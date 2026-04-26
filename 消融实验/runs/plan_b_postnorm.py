"""plan_b_postnorm.py — 回滚到 L1 (Post-Norm + GN) + LR warmup + 长 epochs.

核心假设: v2 L2 (Pre-Norm) 把 M2 从 0.98 打到 0.85, 是 M2 死掉的根因.
Post-Norm 在 v1 (BN) 时代 M2=0.9796, 换成 GN 应该 ≥ 这个水平. 加 warmup + 长 epochs 进一步推到 ≥0.985.

变更 vs L2:
  - pre_norm=False (Post-Norm)
  - epochs=500 (was 300)
  - warmup_epochs=25 (新)
  - 其它 hp 不变 (d=32, do=0.5, wd=1e-4)

阶梯 (任一 IDEAL 即停):
  B1: L1 Post-Norm + ep=500 + warm=25 + (cached d=32 do=0.5 wd=1e-4)  ← 最稳
  B2: B1 + dropout=0.4  (减压让 M2/M5 收敛更快)
  B3: B1 + dropout=0.35 + wd=5e-4
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

# 加载 cached final_hp, 但强制改成 L1 (Post-Norm)
final_hp_cache = load_json(RES / "grid_m6_v2" / "final_hp.json")
BASE_HP = dict(final_hp_cache["final_hp"])
BASE_HP["pre_norm"] = False  # ← 关键改动: L2 → L1
BASE_HP["use_pe"] = False
BASE_HP["n_agents"] = 8
BASE_HP["n_heads"] = 4
LEVEL = "L1"  # Post-Norm + GN


def attack_log_section(title):
    sep = "=" * 78
    block = f"\n\n{sep}\n## PLAN B (Post-Norm): {title}\n{sep}\n"
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
        name="B1_postnorm_warm25_ep500",
        hp_override=dict(epochs=500, warmup_epochs=25),
    ),
    dict(
        name="B2_postnorm_do40_warm25_ep500",
        hp_override=dict(epochs=500, warmup_epochs=25, dropout=0.40),
    ),
    dict(
        name="B3_postnorm_do35_wd5e4_warm30_ep500",
        hp_override=dict(epochs=500, warmup_epochs=30, dropout=0.35,
                          weight_decay=5e-4),
    ),
]


def main():
    t0 = time.time()
    attack_log_section(f"START — base d={BASE_HP['d_model']} do={BASE_HP['dropout']} "
                        f"wd={BASE_HP['weight_decay']:.0e} pre_norm=False level={LEVEL}")
    ov.log("假设: Pre-Norm 是 M2 (mhsa+plain) 塌缩根因. 回滚 Post-Norm 复活 M2.")
    ov.log("基准: v1 BN+Post-Norm M2=0.9796, GN+Post-Norm 应 >=, +warmup +长 ep 再推到 0.985+")

    history = []
    final_winner = None

    for atk in ATTACKS:
        attack_log_section(f"Attack {atk['name']}")
        hp = make_hp(**atk["hp_override"])
        ov.log(f"HP: d={hp['d_model']} do={hp['dropout']} wd={hp['weight_decay']:.0e} "
                f"ep={hp['epochs']} warmup_ep={hp['warmup_epochs']} "
                f"pre_norm={hp['pre_norm']} lr={hp['lr']}")
        try:
            summary = ov.stage_4_full_ablation(hp, LEVEL, label=f"planb_{atk['name']}")
            verdict, means = ov.stage_5_check_and_report(summary, label=f"planb_{atk['name']}")
            entry = dict(name=atk["name"], hp=hp, verdict=verdict,
                         means={k: round(v, 4) for k, v in means.items()})
            history.append(entry)
            save_json(history, RES / "plan_b_postnorm_history.json")

            if verdict.startswith("IDEAL"):
                ov.log(f"[IDEAL HIT] Plan B {atk['name']} -> {verdict}, Plan B 结束")
                final_winner = entry
                break
            else:
                ov.log(f"[NOT IDEAL] Plan B {atk['name']} -> {verdict}, 进下一个")
        except Exception as e:
            ov.log(f"ERROR Plan B {atk['name']}: {e}\n{traceback.format_exc()}")
            history.append(dict(name=atk["name"], hp=hp, error=str(e)))
            save_json(history, RES / "plan_b_postnorm_history.json")

    elapsed_h = (time.time() - t0) / 3600
    if final_winner:
        attack_log_section(f"PLAN B ALL DONE — IDEAL hit by {final_winner['name']} "
                           f"verdict={final_winner['verdict']} elapsed={elapsed_h:.2f}h")
    else:
        attack_log_section(f"PLAN B ALL DONE — 3 轮无 IDEAL, "
                           f"elapsed={elapsed_h:.2f}h. (下一步: 上 Plan C LayerScale)")


if __name__ == "__main__":
    main()
