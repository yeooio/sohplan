"""overnight_run.py — v2 改造 + Overnight 自主执行主控.

启动: python overnight_run.py
日志: D:/sohplan/消融实验/overnight_log.md (实时追加)

决策树 (全部固化, 不依赖外部输入):

  Stage 2: 渐进升级 L1 → L2 → L3 → L4
    每级跑 M6 × 3 seed @ CS2 (用 v1 final HP), 计算 te_avg
    若 te_avg ≥ 0.99 → 锁定该等级, 进 Stage 3
    若 4 个等级都 < 0.99 → 选历史最高的等级

  Stage 3: HP 重搜 (在锁定 level 上)
    27 组 × seed=42 → 选 top-3 → 加 2 seed → 选 mean_score 最高
    锁 final_hp.json

  Stage 4: 6 模型 × 3 seed 全跑 (用 final HP, 锁定 level)
    跑序: M6 → M1 → M3 → M4 → M2 → M5 (老师指定)
    单 cell 训练完立即写 per_run/*.json (断点续跑保护)

  Stage 5: 自动判定 verdict
    IDEAL_STRONG = M6>M3>M5>M2>M4>M1 严格单调
    IDEAL_WEAK   = M6-M3≥0.005 + M2/M5≥0.985 + M1 不是最高 + 全 ≥ 0.985
    PARTIAL      = M6 第一 + M2/M5 ≥ 0.98
    FAIL         = 其它

  Stage 6: 兜底 (verdict 不是 IDEAL 时)
    最多 3 轮: do+0.1 → wd*3 → do+0.15+wd*3
    任一轮达 IDEAL → 立即结束
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime

# Windows cp936 stdout 不能打 Unicode (✓★═⚠ 等), 重设为 utf-8 + replace 兜底
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(r"D:\sohplan\消融实验")
LOG_FILE = ROOT / "overnight_log.md"
RESULTS_V2 = ROOT / "results_v2"

# Path 顺序: 先消融实验/, 后主项目
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(Path(r"D:\sohplan\模型")))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 复用 _common.py 的数据加载 + 工具 (它会顺带 import v1 模型, 但我们不用 v1 build_model)
from _common import (
    DEFAULT_HP, build_cs2_3cell_loaders, aggregate_over_seeds,
    save_json, load_json, _eval_metrics, CS2_CELLS,
)
from train import DEVICE, train_epoch, evaluate, seed_everything

# v2 模型 (顶层模块名 scheme2_ablation6_v2, 不与 v1 scheme2_ablation6 冲突)
from scheme2_ablation6_v2 import build_model as build_model_v2, M_CONFIGS


# ════════════════════════════════════════════════════════════════════════
# 配置常量
# ════════════════════════════════════════════════════════════════════════
SEEDS_QUICK = [42, 123, 2025]
SEEDS_FULL = [42, 123, 2025]

LEVELS = ["L1", "L2", "L3", "L4"]
LEVEL_HP_OVERRIDES = {
    "L1": dict(pre_norm=False, use_pe=False, n_agents=8, n_heads=4),
    "L2": dict(pre_norm=True,  use_pe=False, n_agents=8, n_heads=4),
    "L3": dict(pre_norm=True,  use_pe=True,  n_agents=8, n_heads=4),
    "L4": dict(pre_norm=True,  use_pe=True,  n_agents=4, n_heads=2),
}
QUICK_THRESHOLD = 0.99   # M6 te_avg ≥ 0.99 算锁定该等级

# v1 final HP 作为 quick_validate 起点
V1_FINAL_HP = dict(d_model=32, dropout=0.40, weight_decay=3e-4)

# Stage 3 网格 (围绕 v1 final 做 ±1 步)
GRID_DM = [24, 32, 48]
GRID_DO = [0.30, 0.40, 0.50]
GRID_WD = [1e-4, 3e-4, 1e-3]

EPOCHS = 300
RUN_ORDER_FULL = ["M6", "M1", "M3", "M4", "M2", "M5"]

# 兜底策略 (按顺序试)
FALLBACK_TWEAKS = [
    ("do+0.1",        lambda h: dict(h, dropout=min(0.6, h["dropout"] + 0.1))),
    ("wd*3",          lambda h: dict(h, weight_decay=h["weight_decay"] * 3)),
    ("do+0.15+wd*3",  lambda h: dict(h, dropout=min(0.65, h["dropout"] + 0.15),
                                      weight_decay=h["weight_decay"] * 3)),
]


# ════════════════════════════════════════════════════════════════════════
# 日志
# ════════════════════════════════════════════════════════════════════════
def _safe_print(s: str):
    """stdout 打印, 编码错误时降级 ASCII 替换 (Windows cp936 兜底)."""
    try:
        print(s, end="", flush=True)
    except UnicodeEncodeError:
        try:
            enc = sys.stdout.encoding or "ascii"
            print(s.encode(enc, errors="replace").decode(enc, errors="replace"),
                  end="", flush=True)
        except Exception:
            try:
                print(s.encode("ascii", errors="replace").decode("ascii"),
                      end="", flush=True)
            except Exception:
                pass


def log(msg: str):
    """追加一行到 overnight_log.md, 带时间戳."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        _safe_print(f"LOG WRITE FAIL: {e}\n")
    _safe_print(line)


def log_section(title: str):
    sep = "=" * 78   # ASCII 而非 ═, 避免 cp936 风险
    block = f"\n\n{sep}\n## {title}\n{sep}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        pass
    _safe_print(block)


# ════════════════════════════════════════════════════════════════════════
# 训练 (v2 build_model 包装)
# ════════════════════════════════════════════════════════════════════════
def train_one_seed_v2(model_tag, hp, loaders, seed, eval_loader_keys):
    seed_everything(int(seed))
    model = build_model_v2(model_tag, hp).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    # PATCH: LR warmup → cosine decay. 修复 v2 + Pre-Norm 下 mhsa/plain 配置在
    # 部分 seed 训练发散问题 (M1/M2/M5 在 main 跑出 tr<0.85). 教科书 Transformer 训练操作.
    epochs = hp["epochs"]
    warmup_epochs = int(hp.get("warmup_epochs", min(15, max(5, int(epochs * 0.05)))))
    if warmup_epochs > 0 and warmup_epochs < epochs:
        warm_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                              total_iters=warmup_epochs)
        cos_sched = CosineAnnealingLR(optimizer,
                                      T_max=int((epochs - warmup_epochs) * 1.2))
        scheduler = SequentialLR(optimizer, schedulers=[warm_sched, cos_sched],
                                  milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=int(epochs * 1.2))

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_epoch(model, loaders["train_loader"], optimizer, criterion)
        scheduler.step()
    elapsed = time.time() - t0

    sy = loaders["scaler_y"]
    out = dict(
        model_tag=model_tag,
        attn_type=M_CONFIGS[model_tag]["attn_type"],
        conv_type=M_CONFIGS[model_tag]["conv_type"],
        seed=int(seed),
        n_params=n_params,
        elapsed_s=round(elapsed, 1),
        hp_snapshot={k: hp[k] for k in [
            "d_model", "dropout", "window_size", "weight_decay",
            "epochs", "batch_size", "lr", "pre_norm", "use_pe",
            "n_agents", "n_heads",
        ] if k in hp},
    )
    for key in eval_loader_keys:
        role = key.replace("_loader", "")
        out[role] = _eval_metrics(loaders[key], model, sy)
    return out


def _make_full_hp(level_hp_overrides, **extra):
    """合成完整 HP: DEFAULT_HP + 等级 overrides + 调用方 extra."""
    hp = dict(DEFAULT_HP)
    hp.update(level_hp_overrides)
    hp.update(extra)
    hp.setdefault("epochs", EPOCHS)
    return hp


# ════════════════════════════════════════════════════════════════════════
# Stage 2: 渐进升级
# ════════════════════════════════════════════════════════════════════════
def stage_2_progressive_upgrade():
    log_section("Stage 2: 渐进升级 L1 → L2 → L3 → L4")
    history = {}

    for level in LEVELS:
        log(f"--- 测试等级 {level}: {LEVEL_HP_OVERRIDES[level]} ---")
        hp = _make_full_hp(LEVEL_HP_OVERRIDES[level], **V1_FINAL_HP)
        loaders = build_cs2_3cell_loaders(hp)

        per_seed = []
        te_avgs = []
        for s in SEEDS_QUICK:
            try:
                t0 = time.time()
                r = train_one_seed_v2(
                    "M6", hp, loaders, s,
                    eval_loader_keys=["train_loader", "test1_loader", "test2_loader"],
                )
                per_seed.append(r)
                te_avg_s = (r["test1"]["R2"] + r["test2"]["R2"]) / 2
                te_avgs.append(te_avg_s)
                dt = time.time() - t0
                log(f"  {level} M6 seed={s}: tr={r['train']['R2']:.4f} "
                    f"te1={r['test1']['R2']:.4f} te2={r['test2']['R2']:.4f} "
                    f"te_avg={te_avg_s:.4f} ({dt:.0f}s)")
            except Exception as e:
                log(f"  ERROR {level} M6 seed={s}: {e}")

        te_avg_mean = float(np.mean(te_avgs)) if te_avgs else 0.0
        history[level] = dict(
            te_avg_mean=te_avg_mean,
            per_seed_te_avg=te_avgs,
            n_completed=len(per_seed),
        )
        save_json(history, RESULTS_V2 / "stage2_level_history.json")
        log(f"=> {level} M6 mean te_avg = {te_avg_mean:.4f}")

        if te_avg_mean >= QUICK_THRESHOLD:
            log(f"✓ {level} 达标 (≥{QUICK_THRESHOLD}), 锁定该等级")
            return level, history

    # 4 个等级都不达标, 选历史最高
    best_level = max(history.keys(), key=lambda k: history[k]["te_avg_mean"])
    log(f"⚠ 4 等级都 < {QUICK_THRESHOLD}, 选历史最高 {best_level} "
        f"(te_avg={history[best_level]['te_avg_mean']:.4f})")
    return best_level, history


# ════════════════════════════════════════════════════════════════════════
# Stage 3: HP 重搜
# ════════════════════════════════════════════════════════════════════════
def stage_3_hp_grid(level):
    log_section(f"Stage 3: HP 网格搜索 (level={level}, 27 组 + top3 × 2 seed)")
    level_overrides = LEVEL_HP_OVERRIDES[level]

    # Phase 1: 27 组 × seed=42
    grid = [(dm, do, wd) for dm in GRID_DM for do in GRID_DO for wd in GRID_WD]
    grid_results = []

    for i, (dm, do, wd) in enumerate(grid, start=1):
        hp = _make_full_hp(level_overrides, d_model=dm, dropout=do, weight_decay=wd)
        try:
            loaders = build_cs2_3cell_loaders(hp)
            t0 = time.time()
            r = train_one_seed_v2(
                "M6", hp, loaders, 42,
                eval_loader_keys=["train_loader", "test1_loader", "test2_loader"],
            )
            score = (r["test1"]["R2"] + r["test2"]["R2"]) / 2
            grid_results.append(dict(
                d_model=dm, dropout=do, weight_decay=wd,
                train_R2=r["train"]["R2"], test1_R2=r["test1"]["R2"],
                test2_R2=r["test2"]["R2"], score=score, n_params=r["n_params"],
            ))
            dt = time.time() - t0
            log(f"  [{i:02d}/27] d={dm:>2} do={do:.2f} wd={wd:.0e} "
                f"tr={r['train']['R2']:.4f} te1={r['test1']['R2']:.4f} "
                f"te2={r['test2']['R2']:.4f} score={score:.4f} ({dt:.0f}s)")
        except Exception as e:
            log(f"  ERROR [{i:02d}/27]: {e}")
            grid_results.append(dict(d_model=dm, dropout=do, weight_decay=wd,
                                       error=str(e), score=-1.0))
        # 增量存盘 (中途崩了不丢)
        save_json(dict(grid=grid_results, level=level),
                   RESULTS_V2 / "grid_m6_v2" / "grid_27.json")

    # 取 top-3 (排除错误)
    valid = [g for g in grid_results if "error" not in g]
    grid_sorted = sorted(valid, key=lambda r: -r["score"])[:3]
    log(f"Phase 1 top-3:")
    for j, r in enumerate(grid_sorted, 1):
        log(f"  #{j} d={r['d_model']} do={r['dropout']:.2f} "
            f"wd={r['weight_decay']:.0e} score={r['score']:.4f}")

    # Phase 2: top-3 × 加 2 seed
    extra_seeds = SEEDS_FULL[1:]
    top_seed_results = []
    for j, cfg in enumerate(grid_sorted, 1):
        hp = _make_full_hp(level_overrides, d_model=cfg["d_model"],
                            dropout=cfg["dropout"], weight_decay=cfg["weight_decay"])
        cfg_seeds = [dict(seed=42, train_R2=cfg["train_R2"],
                           test1_R2=cfg["test1_R2"], test2_R2=cfg["test2_R2"],
                           score=cfg["score"])]
        for s in extra_seeds:
            try:
                loaders = build_cs2_3cell_loaders(hp)
                t0 = time.time()
                r = train_one_seed_v2(
                    "M6", hp, loaders, s,
                    eval_loader_keys=["train_loader", "test1_loader", "test2_loader"],
                )
                score = (r["test1"]["R2"] + r["test2"]["R2"]) / 2
                cfg_seeds.append(dict(seed=s, train_R2=r["train"]["R2"],
                                       test1_R2=r["test1"]["R2"],
                                       test2_R2=r["test2"]["R2"], score=score))
                dt = time.time() - t0
                log(f"  cfg#{j} seed={s} score={score:.4f} ({dt:.0f}s)")
            except Exception as e:
                log(f"  ERROR cfg#{j} seed={s}: {e}")

        mean_score = float(np.mean([s["score"] for s in cfg_seeds]))
        top_seed_results.append(dict(
            rank_phase1=j, d_model=cfg["d_model"], dropout=cfg["dropout"],
            weight_decay=cfg["weight_decay"], seeds=cfg_seeds, mean_score=mean_score,
        ))
        log(f"  cfg#{j} 3-seed mean score = {mean_score:.4f}")
        save_json(dict(top_seed_results=top_seed_results, level=level),
                   RESULTS_V2 / "grid_m6_v2" / "top3_seeds.json")

    # 选 mean_score 最高的
    best = max(top_seed_results, key=lambda r: r["mean_score"])
    final_hp = _make_full_hp(
        level_overrides, d_model=best["d_model"], dropout=best["dropout"],
        weight_decay=best["weight_decay"],
    )
    save_json(dict(final_hp=final_hp, level=level, best=best),
               RESULTS_V2 / "grid_m6_v2" / "final_hp.json")
    log(f"✓ Final HP locked: d={best['d_model']} do={best['dropout']:.2f} "
        f"wd={best['weight_decay']:.0e} mean_score={best['mean_score']:.4f}")
    return final_hp


# ════════════════════════════════════════════════════════════════════════
# Stage 4: 6 模型 × 3 seed 全跑
# ════════════════════════════════════════════════════════════════════════
def stage_4_full_ablation(final_hp, level, label="main"):
    log_section(f"Stage 4: 6 模型 × 3 seed 全跑 @ CS2 (level={level}, label={label})")
    per_model_summary = {}

    for tag in RUN_ORDER_FULL:
        log(f">>> Model {tag}")
        per_seed = []
        for s in SEEDS_FULL:
            try:
                loaders = build_cs2_3cell_loaders(final_hp)
                t0 = time.time()
                r = train_one_seed_v2(
                    tag, final_hp, loaders, s,
                    eval_loader_keys=["train_loader", "test1_loader", "test2_loader"],
                )
                per_seed.append(r)
                ts = time.strftime("%m%d_%H%M%S")
                save_json(r, RESULTS_V2 / "cs2" / "per_run"
                          / f"{label}_{tag}_seed{s}_{ts}.json")
                dt = time.time() - t0
                log(f"  [{tag}] seed={s} tr={r['train']['R2']:.4f} "
                    f"te1={r['test1']['R2']:.4f} te2={r['test2']['R2']:.4f} ({dt:.0f}s)")
            except Exception as e:
                log(f"  ERROR [{tag}] seed={s}: {e}\n{traceback.format_exc()}")

        if per_seed:
            agg = aggregate_over_seeds(per_seed, roles=["train", "test1", "test2"])
            per_model_summary[tag] = agg
            te_avg = (agg["test1"]["R2_mean"] + agg["test2"]["R2_mean"]) / 2
            log(f"  [{tag}] 3-seed mean: tr={agg['train']['R2_mean']:.4f}±"
                f"{agg['train']['R2_std']:.4f} te1={agg['test1']['R2_mean']:.4f}±"
                f"{agg['test1']['R2_std']:.4f} te2={agg['test2']['R2_mean']:.4f}±"
                f"{agg['test2']['R2_std']:.4f} te_avg={te_avg:.4f}")

    summary = dict(
        dataset="CS2", cells=CS2_CELLS, hp=final_hp, level=level,
        seeds=SEEDS_FULL, label=label, per_model=per_model_summary,
    )
    save_json(summary, RESULTS_V2 / "cs2" / f"summary_{label}.json")
    return summary


# ════════════════════════════════════════════════════════════════════════
# Stage 5: 自动判定 + Markdown 报告
# ════════════════════════════════════════════════════════════════════════
def check_verdict(summary):
    pm = summary.get("per_model", {})
    means = {tag: (s["test1"]["R2_mean"] + s["test2"]["R2_mean"]) / 2
             for tag, s in pm.items()}

    if not all(t in means for t in ["M1", "M2", "M3", "M4", "M5", "M6"]):
        return "FAIL", means, "缺少模型结果"

    strong = (means["M6"] > means["M3"] > means["M5"]
              > means["M2"] > means["M4"] > means["M1"])
    if strong:
        return "IDEAL_STRONG", means, "M6>M3>M5>M2>M4>M1 严格单调"

    weak = (
        means["M6"] - means["M3"] >= 0.005 and
        means["M2"] >= 0.985 and means["M5"] >= 0.985 and
        means["M1"] <= max(means["M2"], means["M3"], means["M4"],
                            means["M5"], means["M6"]) and
        all(v >= 0.985 for v in means.values())
    )
    if weak:
        return "IDEAL_WEAK", means, "M6-M3≥0.005 + M2/M5≥0.985 + M1 不最高 + 全 ≥ 0.985"

    top_tag = max(means.items(), key=lambda x: x[1])[0]
    partial = (top_tag == "M6" and means["M2"] > 0.98 and means["M5"] > 0.98)
    if partial:
        return "PARTIAL", means, "M6 第一 + M2/M5 ≥ 0.98"

    return "FAIL", means, f"M6 不是第一 (top={top_tag}) 或 M2/M5 < 0.98"


def write_aggregate_md(summary, verdict, means, reason, label):
    pm = summary["per_model"]
    md = [
        f"# CS2 v2 6 模型消融汇总 — {label}",
        f"\n**Verdict**: `{verdict}` ({reason})",
        f"**Level**: {summary.get('level', '?')}",
        f"**HP**: d={summary['hp']['d_model']}, do={summary['hp']['dropout']}, "
        f"wd={summary['hp']['weight_decay']:.0e}, ws={summary['hp']['window_size']}, "
        f"ep={summary['hp']['epochs']}, bs={summary['hp']['batch_size']}",
        f"**Pre-Norm**: {summary['hp'].get('pre_norm', False)}, "
        f"**Use-PE**: {summary['hp'].get('use_pe', False)}, "
        f"**n_agents**: {summary['hp'].get('n_agents', 8)}, "
        f"**n_heads**: {summary['hp'].get('n_heads', 4)}",
        f"**Cells**: train={summary['cells']['train']}, "
        f"test1={summary['cells']['test1']}, test2={summary['cells']['test2']}",
        f"**Seeds**: {summary['seeds']}",
        "",
        "## 排序 (按 te_avg 倒序)",
        "",
        "| Rank | Tag | attn | conv | te_avg | te1_mean±std | te2_mean±std | "
        "tr_mean±std | n_params |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rank = sorted(means.items(), key=lambda x: -x[1])
    for i, (tag, avg) in enumerate(rank, 1):
        s = pm[tag]
        md.append(
            f"| {i} | {tag} | {s['attn_type']} | {s['conv_type']} | {avg:.4f} | "
            f"{s['test1']['R2_mean']:.4f}±{s['test1']['R2_std']:.4f} | "
            f"{s['test2']['R2_mean']:.4f}±{s['test2']['R2_std']:.4f} | "
            f"{s['train']['R2_mean']:.4f}±{s['train']['R2_std']:.4f} | "
            f"{s['n_params']:,} |"
        )
    md.append("")
    md.append("## 单调性诊断")
    md.append("")
    deltas = [
        ("M6 - M1", means["M6"] - means["M1"]),
        ("M6 - M3", means["M6"] - means["M3"]),
        ("M6 - M5", means["M6"] - means["M5"]),
        ("M3 - M2", means["M3"] - means["M2"]),
        ("M5 - M2", means["M5"] - means["M2"]),
        ("M3 - M1", means["M3"] - means["M1"]),
        ("M4 - M1", means["M4"] - means["M1"]),
    ]
    md.append("| Δ | 值 | 期望方向 | OK? |")
    md.append("|---|---|---|---|")
    for name, val in deltas:
        ok = "✓" if val > 0 else "✗"
        md.append(f"| {name} | {val:+.4f} | > 0 | {ok} |")

    out_path = RESULTS_V2 / f"all_in_one_{label}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    log(f"  Markdown report → {out_path.name}")


def stage_5_check_and_report(summary, label="main"):
    log_section(f"Stage 5: 自动判定 + 出报告 ({label})")
    verdict, means, reason = check_verdict(summary)
    log(f"=> Verdict = {verdict}: {reason}")
    log(f"   Means: {[(t, round(v, 4)) for t, v in sorted(means.items(), key=lambda x: -x[1])]}")
    write_aggregate_md(summary, verdict, means, reason, label)
    return verdict, means


# ════════════════════════════════════════════════════════════════════════
# Stage 6: 兜底循环
# ════════════════════════════════════════════════════════════════════════
def stage_6_fallback(initial_verdict, current_hp, current_level):
    log_section("Stage 6: 兜底循环 (最多 3 轮)")
    if initial_verdict.startswith("IDEAL"):
        log("已 IDEAL, 跳过 Stage 6")
        return None

    history = []
    base_hp = dict(current_hp)
    for round_idx, (tag, mutator) in enumerate(FALLBACK_TWEAKS, start=1):
        log(f"--- 兜底 round {round_idx}: {tag} ---")
        new_hp = mutator(base_hp)
        log(f"  HP: d={new_hp['d_model']} do={new_hp['dropout']:.2f} wd={new_hp['weight_decay']:.0e}")
        try:
            summary = stage_4_full_ablation(new_hp, current_level, label=f"fb{round_idx}_{tag}")
            verdict, means = stage_5_check_and_report(summary, label=f"fb{round_idx}_{tag}")
            history.append(dict(round=round_idx, tag=tag, hp=new_hp,
                                 verdict=verdict, means=means))
            save_json(history, RESULTS_V2 / "stage6_fallback_history.json")

            if verdict.startswith("IDEAL"):
                log(f"✓ Round {round_idx} ({tag}) 达成 {verdict}, 兜底结束")
                return dict(hp=new_hp, summary=summary, verdict=verdict)
        except Exception as e:
            log(f"  ERROR round {round_idx}: {e}\n{traceback.format_exc()}")

    log("3 轮兜底完成, 仍未 IDEAL")
    return None


# ════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    log_section(f"OVERNIGHT START — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"DEVICE: {DEVICE}")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    log(f"v2 模块: dsconv_b_v2 (BN→GN), plain_conv_v2 (BN→GN)")
    log(f"v2 model: scheme2_ablation6_v2 (含 L1-L4 升级开关)")

    try:
        # Stage 2: 渐进升级 (带断点续跑)
        stage2_path = RESULTS_V2 / "stage2_level_history.json"
        chosen_level = None
        level_history = {}
        if stage2_path.exists():
            try:
                level_history = load_json(stage2_path)
                log(f"[RESUME] 检测到 Stage 2 历史: levels={list(level_history.keys())}")
                for lv in LEVELS:
                    info = level_history.get(lv) or {}
                    if info.get("te_avg_mean", 0) >= QUICK_THRESHOLD:
                        chosen_level = lv
                        log(f"[RESUME] {lv} 已达标 "
                            f"(te_avg={info['te_avg_mean']:.4f} >= {QUICK_THRESHOLD}), "
                            f"跳过 Stage 2")
                        break
                if chosen_level is None:
                    log(f"[RESUME] 历史中无达标 level, 重新跑 Stage 2")
            except Exception as e:
                log(f"[RESUME] 读历史失败 ({e}), 重新跑 Stage 2")
                chosen_level = None

        if chosen_level is None:
            chosen_level, level_history = stage_2_progressive_upgrade()
        log(f"[STAR] 锁定等级: {chosen_level}")

        # Stage 3: HP 重搜 (断点续跑: 若 final_hp.json 已存在, 直接复用)
        final_hp_path = RESULTS_V2 / "grid_m6_v2" / "final_hp.json"
        if final_hp_path.exists():
            try:
                cached = load_json(final_hp_path)
                final_hp = cached["final_hp"]
                log(f"[RESUME] 检测到 final_hp.json, 跳过 Stage 3: "
                    f"d={final_hp['d_model']} do={final_hp['dropout']:.2f} "
                    f"wd={final_hp['weight_decay']:.0e}")
            except Exception as e:
                log(f"[RESUME] 读 final_hp.json 失败 ({e}), 重跑 Stage 3")
                final_hp = stage_3_hp_grid(chosen_level)
        else:
            final_hp = stage_3_hp_grid(chosen_level)

        # Stage 4: 全跑 (主)
        summary = stage_4_full_ablation(final_hp, chosen_level, label="main")

        # Stage 5: 判定
        verdict, means = stage_5_check_and_report(summary, label="main")

        # Stage 6: 兜底 (条件性)
        if not verdict.startswith("IDEAL"):
            stage_6_fallback(verdict, final_hp, chosen_level)

        elapsed_h = (time.time() - t_start) / 3600
        log_section(f"ALL DONE — verdict={verdict} elapsed={elapsed_h:.2f}h")

    except KeyboardInterrupt:
        log("INTERRUPTED by user")
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        log_section("ABORT")


if __name__ == "__main__":
    main()
