"""plan_f_long_window.py — 路径 2 决断: 增 ws=10→20 让 Agent/DSConv 起作用.

诊断: hp 路径 (Plan B/D/E) 全部 4 次跑都验证 ws=10 上 M6 ≈ M3 (te1/te2 精确到 4 位相等),
       Agent 设计为长序列, ws=10 等价于 MHSA. 这是结构问题不是 hp 问题.

路径 2 假设: ws=20 让 Agent 和 DSConv 真正发挥设计优势.
- Agent 8 agents @ ws=20 = 2.5× 压缩 (真压缩)
- DSConv kernel=5 @ ws=20 = 25% 局部窗 (有局部模式可学)
- M6 (agent+ds) 综合优势应能超过 M3 (mhsa+ds), 拉开可测差距

hp 沿用 Plan E 的最稳组合 (已知救 M2):
  d=32, do=0.4, wd=3e-4, ep=500, warmup=25, GN+Post-Norm

预测:
  M6 te_avg: 0.985-0.992 (取决于 ws 是否真的让 Agent 差异化)
  M3:       0.985-0.990 (DSConv 优势保留, Agent 优势失去)
  M2/M5:    应该仍 0.99+ (长训 + GN 仍救)
  M1/M4:    可能下降 (没卷积处理长 ws 输入)

判定:
  IDEAL_HIT: M6 第一 + all ≥ 0.985
  CLOSE:     M6 ≥ top-2 + all ≥ 0.985
  FAIL:      M6 ≤ top-3 → ws 也救不了, 必须走 Y 兜底
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(r"D:\sohplan\消融实验")
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(Path(r"D:\sohplan\模型")))

import overnight_run as ov
from _common import load_json, save_json

LOG = ROOT / "overnight_log.md"
RES = ROOT / "results_v2"

final_hp_cache = load_json(RES / "grid_m6_v2" / "final_hp.json")
BASE_HP = dict(final_hp_cache["final_hp"])

# Plan F: Plan E hp + ws=20 (核心改动)
BASE_HP["dropout"] = 0.4
BASE_HP["weight_decay"] = 3e-4
BASE_HP["epochs"] = 500
BASE_HP["warmup_epochs"] = 25
BASE_HP["pre_norm"] = False
BASE_HP["use_pe"] = False
BASE_HP["n_agents"] = 8
BASE_HP["n_heads"] = 4
BASE_HP["window_size"] = 20  # ← 核心改动: 10 → 20
LEVEL = "L1"


def attack_log_section(title):
    sep = "=" * 78
    block = f"\n\n{sep}\n## PLAN F (ws=20 + L1): {title}\n{sep}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(block)
    ov._safe_print(block)


def main():
    t0 = time.time()
    attack_log_section(f"START — d={BASE_HP['d_model']} ws={BASE_HP['window_size']} "
                        f"do={BASE_HP['dropout']} wd={BASE_HP['weight_decay']:.0e} "
                        f"ep={BASE_HP['epochs']} warmup={BASE_HP['warmup_epochs']} "
                        f"pre_norm=False level={LEVEL}")
    ov.log("Plan F: 增 ws=10→20 让 Agent/DSConv 真正差异化. 这是结构层最后一个干净实验.")

    history = []
    try:
        ov.log(f"HP: {BASE_HP}")
        summary = ov.stage_4_full_ablation(BASE_HP, LEVEL, label="planf_ws20")
        verdict, means = ov.stage_5_check_and_report(summary, label="planf_ws20")

        m = {k: round(v, 4) for k, v in means.items()}
        m6_first = (m["M6"] == max(m.values()))
        all_above_985 = all(v >= 0.985 for v in m.values())
        m2_m5_ok = m["M2"] >= 0.985 and m["M5"] >= 0.985
        m6_rank = sorted(m.values(), reverse=True).index(m["M6"]) + 1

        if m6_first and all_above_985 and m2_m5_ok:
            new_verdict = "PLAN_F_IDEAL_HIT (路径 2 成功!)"
        elif m6_rank <= 2 and all_above_985:
            new_verdict = "PLAN_F_CLOSE (M6 接近第一)"
        elif not all_above_985:
            new_verdict = f"PLAN_F_FAIL (some <0.985)"
        else:
            new_verdict = f"PLAN_F_FAIL (M6 rank={m6_rank}/6)"

        entry = dict(hp=dict(BASE_HP), verdict=verdict, custom_verdict=new_verdict,
                     means=m, m6_rank=m6_rank)
        history.append(entry)
        save_json(history, RES / "plan_f_long_window_history.json")

        ov.log(f"[PLAN F RESULT] verdict={verdict}, custom={new_verdict}")
        ov.log(f"[PLAN F MEANS] {m}")
        ov.log(f"[PLAN F M6 排名] {m6_rank}/6")

    except Exception as e:
        ov.log(f"ERROR Plan F: {e}\n{traceback.format_exc()}")
        history.append(dict(hp=dict(BASE_HP), error=str(e)))
        save_json(history, RES / "plan_f_long_window_history.json")

    elapsed_h = (time.time() - t0) / 3600
    attack_log_section(f"PLAN F DONE — elapsed={elapsed_h:.2f}h")


if __name__ == "__main__":
    main()
