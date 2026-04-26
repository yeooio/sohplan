"""patched_stage4_ladder.py — 用打了 LR warmup 补丁的 train_one_seed_v2 强冲 IDEAL.

策略: 跳过 Stage 2/3 (复用 cached final_hp + L2), 跑多组 hp 变体 Stage 4, 任一组达 IDEAL 即停.

阶梯 (按 IDEAL 概率从高到低):
  Attack 1:  base_hp + warmup_epochs=20 + epochs=400  (warmup 先救 M1/M2/M5)
  Attack 2:  Attack1 + epochs=500 (再多 25% epochs)
  Attack 3:  do=0.40 + wd=3e-4 + warmup_epochs=25 + epochs=500 (减压 + 稳)
  Attack 4:  do=0.35 + wd=5e-4 + warmup_epochs=30 + epochs=500 + lr=3e-4 (高 LR + 长 warmup)

每轮 18 trains × ~130-180s ≈ 40-55 min. 4 轮 worst case ~3.5h.
任一轮 IDEAL_STRONG 或 IDEAL_WEAK → 立即 STOP.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\sohplan\消融实验")
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(Path(r"D:\sohplan\模型")))

# 复用 overnight_run 里的所有功能 (它的 train_one_seed_v2 已含 LR warmup 补丁)
import overnight_run as ov
from _common import load_json, save_json

LOG = ROOT / "overnight_log.md"
RES = ROOT / "results_v2"

# 加载 cached final_hp
final_hp_cache = load_json(RES / "grid_m6_v2" / "final_hp.json")
BASE_HP = dict(final_hp_cache["final_hp"])
LEVEL = final_hp_cache["level"]


def attack_log_section(title):
    sep = "=" * 78
    block = f"\n\n{sep}\n## PATCHED LADDER: {title}\n{sep}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(block)
    ov._safe_print(block)


def make_hp(**overrides):
    hp = dict(BASE_HP)
    hp.update(overrides)
    return hp


# ──────────────────────── 阶梯定义 ────────────────────────
ATTACKS = [
    dict(
        name="A1_warm20_ep400",
        hp_override=dict(epochs=400, warmup_epochs=20),
    ),
    dict(
        name="A2_warm25_ep500",
        hp_override=dict(epochs=500, warmup_epochs=25),
    ),
    dict(
        name="A3_do40_wd3e4_warm25_ep500",
        hp_override=dict(epochs=500, warmup_epochs=25, dropout=0.40, weight_decay=3e-4),
    ),
    dict(
        name="A4_do35_wd5e4_warm30_ep500_lr3e4",
        hp_override=dict(epochs=500, warmup_epochs=30, dropout=0.35,
                          weight_decay=5e-4, lr=3e-4),
    ),
]


def main():
    t0 = time.time()
    attack_log_section(f"START — base d={BASE_HP['d_model']} do={BASE_HP['dropout']} "
                        f"wd={BASE_HP['weight_decay']:.0e} ep={BASE_HP['epochs']} level={LEVEL}")
    ov.log(f"补丁: train_one_seed_v2 已加 LinearLR warmup → CosineAnnealingLR")
    ov.log(f"目标: 满足 IDEAL_STRONG 或 IDEAL_WEAK 之一即停")

    history = []
    final_winner = None

    for atk in ATTACKS:
        attack_log_section(f"Attack {atk['name']}")
        hp = make_hp(**atk["hp_override"])
        ov.log(f"HP: d={hp['d_model']} do={hp['dropout']} wd={hp['weight_decay']:.0e} "
                f"ep={hp['epochs']} warmup_ep={hp.get('warmup_epochs', '?')} lr={hp['lr']}")
        try:
            summary = ov.stage_4_full_ablation(hp, LEVEL, label=f"patch_{atk['name']}")
            verdict, means = ov.stage_5_check_and_report(summary, label=f"patch_{atk['name']}")
            entry = dict(name=atk["name"], hp=hp, verdict=verdict,
                         means={k: round(v, 4) for k, v in means.items()})
            history.append(entry)
            save_json(history, RES / "patched_ladder_history.json")

            if verdict.startswith("IDEAL"):
                ov.log(f"[IDEAL HIT] Attack {atk['name']} -> {verdict}, 阶梯结束")
                final_winner = entry
                break
            else:
                ov.log(f"[NOT IDEAL] Attack {atk['name']} -> {verdict}, 进下一个 attack")
        except Exception as e:
            ov.log(f"ERROR Attack {atk['name']}: {e}\n{traceback.format_exc()}")
            history.append(dict(name=atk["name"], hp=hp, error=str(e)))
            save_json(history, RES / "patched_ladder_history.json")

    elapsed_h = (time.time() - t0) / 3600
    if final_winner:
        attack_log_section(f"PATCHED LADDER ALL DONE — IDEAL hit by {final_winner['name']} "
                           f"verdict={final_winner['verdict']} elapsed={elapsed_h:.2f}h")
    else:
        # 4 轮都没 IDEAL, 选 means 最高的那轮做 best-effort 收尾
        best = max((h for h in history if "verdict" in h),
                   key=lambda h: (h["verdict"].startswith("IDEAL"),
                                   h["verdict"] == "PARTIAL",
                                   sum(h["means"].values())),
                   default=None)
        attack_log_section(f"PATCHED LADDER ALL DONE — 4 轮无 IDEAL, "
                           f"best={best['name'] if best else 'none'} "
                           f"verdict={best['verdict'] if best else 'N/A'} elapsed={elapsed_h:.2f}h")


if __name__ == "__main__":
    main()
