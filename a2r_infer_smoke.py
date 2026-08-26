#!/usr/bin/env python3
"""A2R 推理端冒烟测试：不需要仿真器，直接验证「读 -> 跑 -> 写」这个闭环。

它做的事：建策略 -> 灌 16 帧假观测 -> infer -> 打印滚动堆的状态，重复几轮。
要看的是：不崩、每轮堆在长、覆盖的帧数在涨、动作形状正常。

  uv run python a2r_infer_smoke.py --ckpt runs/ckpts/mme_vla_suite/a2r_union/79999
"""
import argparse, pathlib
import numpy as np

P = argparse.ArgumentParser()
P.add_argument("--ckpt", default="runs/ckpts/mme_vla_suite/a2r_union/79999")
P.add_argument("--rounds", type=int, default=5)
P.add_argument("--horizon", type=int, default=16)   # 每轮灌多少帧（= obs_horizon）
A = P.parse_args()

import mme_vla_suite.training.config as _config
from mme_vla_suite.policies import policy_config as _pc

ck = pathlib.Path(A.ckpt)
hcp = ck.parent / "history_config.txt"
print("checkpoint     :", ck)
print("history_config :", hcp.read_text().strip() if hcp.exists() else "(缺失!)")

tc = _config.get_config("mme_vla_suite")
print("\n加载策略 ...", flush=True)
policy = _pc.create_trained_policy(tc, ck, seed=7)
sel = getattr(policy, "a2r_selector", None)
print("a2r_selector   :", "已建立" if sel is not None else "!! 是 None，说明没走 a2r 分支")
if sel is None:
    raise SystemExit("配置不是 a2r，这个测试没有意义")
print("budget=%d  n_diff=%d  n_attn=%d  stride=%d  num_views=%d"
      % (sel.budget, sel.n_diff, sel.n_attn, sel.stride, sel.num_views))

rng = np.random.default_rng(0)
H = A.horizon
state_dim = 8

for r in range(A.rounds):
    imgs = rng.integers(0, 255, (H, 1, 224, 224, 3), dtype=np.uint8)
    states = rng.normal(size=(H, state_dim)).astype(np.float32)
    policy.add_buffer({"images": imgs, "state": states, "add_buffer": True,
                       "exec_start_idx": 0})
    obs = {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.normal(size=(state_dim,)).astype(np.float32),
        "prompt": "watch the video carefully, then pick up the container hiding the red cube",
    }
    out = policy.infer(obs)
    st = sel.stats()
    print("轮 %d  step_idx=%-4d  动作 %s  堆: novelty %3d / relevance %3d, 覆盖 %3d 帧  (%.0f ms)"
          % (r + 1, policy.step_idx, np.asarray(out["actions"]).shape,
             st["n_diff"], st["n_attn"], st["n_frames"], out["infer_time_ms"]), flush=True)

print()
print("=== 判据 ===")
print("1. 没崩                      -> 读/跑/写的闭环通了")
print("2. 两个堆的数字在涨          -> observe 确实在写")
print("3. 覆盖帧数随轮次增加        -> 记忆真的跨帧累积，不是只有当前帧")
print("4. 两个堆都在涨          -> stride=%d，每次调用都写 novelty（与训练的帧间隔对齐）" % sel.stride)
