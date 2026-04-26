"""plan_e_combined.py — 组合实验: v1 hp (救 M6) + 长训 (救 M2) + GN+Post-Norm.

诊断: B1 vs Plan D 两次都 FAIL, 但失败原因不同
  B1 (do=0.5, wd=1e-4, ep=500, warm=25): M2 救活但 M6 沉底 (4th)
  Plan D (do=0.4, wd=3e-4, ep=300, warm=15): M6 微弱反弹 (但仍 4th), M2 塌

Plan E 假设: M6 需要 v1 hp (do/wd), M2 需要长训 (ep=500), 两者可以同时满足.

vs Plan D 改的: ep=300 → 500, warmup=15 → 25
vs B1 改的:    do=0.5 → 0.4, wd=1e-4 → 3e-4
最终 hp:
  d=32, do=0.4, wd=3e-4, ep=500, warmup=25, lr=2e-4
  GN, Post-Norm, n_agents=8, n_heads=4

预测 (基于 B1+Plan D 推断):
  M6: 0.9900-0.9905 (B1 0.9886 与 Plan D 0.9900 的延伸)
  M2: 0.99-0.991 (长训应该救回, B1 已证)
  M3: 0.99 (Plan D 显示)
  M4: 0.99
  M5: 0.99
  M1: 0.99

判定:
  IDEAL_HIT: M6 第一 + all ≥ 0.985 + M2/M5 ≥ 0.985 → 论文铁证
  CLOSE:     M6 ≥ top-3 + all ≥ 0.985 → 松绑定义后写
  FAIL:      M6 ≤ top-4 或 M2/M5 塌 → 物理不可达, 走 Y 兜底
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

# Plan E: v1 hp 的 do/wd + B1 的 ep/warmup + L1
BASE_HP["dropout"] = 0.4
BASE_HP["weight_decay"] = 3e-4
BASE_HP["epochs"] = 500
BASE_HP["warmup_epochs"] = 25
BASE_HP["pre_norm"] = False
BASE_HP["use_pe"] = False
BASE_HP["n_agents"] = 8
BASE_HP["n_heads"] = 4
LEVEL = "L1"


def attack_log_section(title):
    sep = "=" * 78
    block = f"\n\n{sep}\n## PLAN E (v1-hp + long-train + L1): {title}\n{sep}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(block)
    ov._safe_print(block)


def main():
    t0 = time.time()
    attack_log_section(f"START — d={BASE_HP['d_model']} do={BASE_HP['dropout']} "
                        f"wd={BASE_HP['weight_decay']:.0e} ep={BASE_HP['epochs']} "
                        f"warmup={BASE_HP['warmup_epochs']} pre_norm=False level={LEVEL}")
    ov.log("Plan E: v1 的 do/wd (救 M6) + 长训 (救 M2). 这是 hp 路径最后一个未测组合.")

    history = []
    try:
        ov.log(f"HP: {BASE_HP}")
        summary = ov.stage_4_full_ablation(BASE_HP, LEVEL, label="plane_combined")
        verdict, means = ov.stage_5_check_and_report(summary, label="plane_combined")

        m = {k: round(v, 4) for k, v in means.items()}
        m6_first = (m["M6"] == max(m.values()))
        all_above_985 = all(v >= 0.985 for v in m.values())
        m2_m5_ok = m["M2"] >= 0.985 and m["M5"] >= 0.985
        m6_rank = sorted(m.values(), reverse=True).index(m["M6"]) + 1

        if m6_first and all_above_985 and m2_m5_ok:
            new_verdict = "PLAN_E_IDEAL_HIT (论文铁证)"
        elif m6_rank <= 3 and all_above_985:
            new_verdict = "PLAN_E_CLOSE (Y-relaxed 定义可用)"
        elif not all_above_985:
            new_verdict = f"PLAN_E_FAIL (some <0.985: 物理不可达)"
        else:
            new_verdict = f"PLAN_E_FAIL (M6 rank={m6_rank}/6)"

        entry = dict(hp=dict(BASE_HP), verdict=verdict, custom_verdict=new_verdict,
                     means=m, m6_rank=m6_rank)
        history.append(entry)
        save_json(history, RES / "plan_e_combined_history.json")

        ov.log(f"[PLAN E RESULT] verdict={verdict}, custom={new_verdict}")
        ov.log(f"[PLAN E MEANS] {m}")
        ov.log(f"[PLAN E M6 排名] {m6_rank}/6")

    except Exception as e:
        ov.log(f"ERROR Plan E: {e}\n{traceback.format_exc()}")
        history.append(dict(hp=dict(BASE_HP), error=str(e)))
        save_json(history, RES / "plan_e_combined_history.json")

    elapsed_h = (time.time() - t0) / 3600
    attack_log_section(f"PLAN E DONE — elapsed={elapsed_h:.2f}h")


if __name__ == "__main__":
    main()
