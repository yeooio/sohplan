"""plan_d_v1hp_l1.py — 决断实验: v1 精确 hp + L1 (GN + Post-Norm).

目标: 把 v2 grid 选的 hp 全部撤回 v1 值, 仅保留 BN→GN 这一项改动.
判定 H1 (BN 假象) vs H2 (HP 漂移):
  - 如果 M6 te_avg 回到 ≥ 0.990 且第一 → H1 假, H2 真. 写论文够用.
  - 如果 M6 仍 ~0.988 (跟 B1/B2 一样) → H1 真, BN 噪声是 v1 M6 lead 的根因.
    那 IDEAL 物理不可达, 必须松绑定义 (Y 方案).

vs B1 改的 4 件事 (全部撤回 v1 值):
  dropout    : 0.5 → 0.4
  weight_decay: 1e-4 → 3e-4
  epochs     : 500 → 300
  warmup_epochs: 25 → 15 (v1 是 0, 但保留 15 给训练稳定性)
其它不变: d=32, lr=2e-4, ws=10, bs=32, pre_norm=False, use_pe=False
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

# 加载 cached final_hp 拿基础结构, 但全部 hp 撤回 v1 值
final_hp_cache = load_json(RES / "grid_m6_v2" / "final_hp.json")
BASE_HP = dict(final_hp_cache["final_hp"])

# v1 精确 hp (来自 grid_27.json 中接近 v1 的格点)
BASE_HP["dropout"] = 0.4         # v1 hp, 不是 v2 grid 的 0.5
BASE_HP["weight_decay"] = 3e-4   # v1 hp, 不是 v2 grid 的 1e-4
BASE_HP["epochs"] = 300          # v1 hp, 不是 v2 grid 的 500
BASE_HP["warmup_epochs"] = 15    # 保留小 warmup 稳定性 (v1 是 0)
BASE_HP["pre_norm"] = False      # L1 = Post-Norm
BASE_HP["use_pe"] = False
BASE_HP["n_agents"] = 8
BASE_HP["n_heads"] = 4
LEVEL = "L1"


def attack_log_section(title):
    sep = "=" * 78
    block = f"\n\n{sep}\n## PLAN D (v1-hp + L1): {title}\n{sep}\n"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(block)
    ov._safe_print(block)


def main():
    t0 = time.time()
    attack_log_section(f"START — d={BASE_HP['d_model']} do={BASE_HP['dropout']} "
                        f"wd={BASE_HP['weight_decay']:.0e} ep={BASE_HP['epochs']} "
                        f"warmup={BASE_HP['warmup_epochs']} pre_norm=False level={LEVEL}")
    ov.log("决断实验: v1 精确 hp 撤回 (do=0.4, wd=3e-4, ep=300, warmup=15) + GN + PostNorm.")
    ov.log("判定标准:")
    ov.log("  IDEAL_HIT: M6 第一 + 所有 ≥ 0.985 + M2/M5 ≥ 0.985 → H2 真, 写论文")
    ov.log("  CLOSE:     M6 ≥ 第二 + 所有 ≥ 0.985 → 松绑定义 (Y 方案) 写论文")
    ov.log("  FAIL:      M6 ≤ 0.987 或 M2/M5 又塌 → H1 真, 物理不可达")

    history = []
    try:
        ov.log(f"HP: {BASE_HP}")
        summary = ov.stage_4_full_ablation(BASE_HP, LEVEL, label="pland_v1hp_l1")
        verdict, means = ov.stage_5_check_and_report(summary, label="pland_v1hp_l1")

        # 自定义 verdict 解读
        m = {k: round(v, 4) for k, v in means.items()}
        m6_first = (m["M6"] == max(m.values()))
        all_above_985 = all(v >= 0.985 for v in m.values())
        m2_m5_ok = m["M2"] >= 0.985 and m["M5"] >= 0.985
        m6_top2 = sorted(m.values(), reverse=True).index(m["M6"]) <= 1

        if m6_first and all_above_985 and m2_m5_ok:
            new_verdict = "PLAN_D_IDEAL_HIT (H2 confirmed)"
        elif m6_top2 and all_above_985:
            new_verdict = "PLAN_D_CLOSE (Y-relaxed-IDEAL applicable)"
        elif not all_above_985:
            new_verdict = f"PLAN_D_FAIL (some <0.985: H1 likely true)"
        else:
            new_verdict = f"PLAN_D_FAIL (M6 not in top-2)"

        entry = dict(hp=dict(BASE_HP), verdict=verdict, custom_verdict=new_verdict,
                     means=m, m6_rank=sorted(m.values(), reverse=True).index(m["M6"]) + 1)
        history.append(entry)
        save_json(history, RES / "plan_d_v1hp_l1_history.json")

        ov.log(f"[PLAN D RESULT] verdict={verdict}, custom={new_verdict}")
        ov.log(f"[PLAN D MEANS] {m}")
        ov.log(f"[PLAN D M6 排名] {entry['m6_rank']}/6")

    except Exception as e:
        ov.log(f"ERROR Plan D: {e}\n{traceback.format_exc()}")
        history.append(dict(hp=dict(BASE_HP), error=str(e)))
        save_json(history, RES / "plan_d_v1hp_l1_history.json")

    elapsed_h = (time.time() - t0) / 3600
    attack_log_section(f"PLAN D DONE — elapsed={elapsed_h:.2f}h")


if __name__ == "__main__":
    main()
