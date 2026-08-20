"""A2R 的选择逻辑：novelty ∪ relevance，各占一半名额。

两条通道都在**同一个未经预筛的候选池**上打分（这是 A2R 成立的前提）：
  novelty   : 这个格子相比上一张候选帧同位置的格子变了多少（TokenDrop 的打分规则）
  relevance : 模型自己的 action→memory 注意力（由 pass A 抓出来）

JAX 下不能像 PyTorch 那样 `unique` 出一个变长结果再补齐（jit 要求形状固定），
所以并集用「布尔掩码 + 加大常数排序」实现，见 union_select。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

NEG = -1e9


def novelty_scores(static_image_emb, static_mask, n_frames, per_frame,
                   protect_frame0=True):
    """(B, n_cand, d) 原始视觉特征 -> (B, n_cand) 的 novelty 分数。

    槽位布局是「帧 f * per_frame + 格子 p」。每个格子减去**上一张候选帧同一位置**的格子，
    逐维取绝对值再平均。第 0 张没有上一张，按 TokenDrop 的哨兵做法给一个极大值，
    保证它必定入选（对应 TokenDrop 里 frame-0 全保留那条规则）。
    无效槽位（右侧补齐出来的）压到最低。
    """
    B, N, d = static_image_emb.shape
    v = static_image_emb.reshape(B, n_frames, per_frame, d).astype(jnp.float32)
    prev = jnp.concatenate([v[:, :1], v[:, :-1]], axis=1)
    nov = jnp.abs(v - prev).mean(axis=-1)                       # (B, n_frames, per_frame)
    if protect_frame0:
        nov = nov.at[:, 0, :].set(1e6)
    nov = nov.reshape(B, N)
    return jnp.where(static_mask, nov, NEG)


def _topk_mask(scores, k, n):
    """(B, n) 分数 -> (B, n) 布尔，前 k 名为 True。"""
    idx = jnp.argsort(scores, axis=-1)[:, ::-1][:, :k]          # (B, k)
    return jnp.zeros_like(scores, dtype=bool).at[
        jnp.arange(scores.shape[0])[:, None], idx].set(True)


def union_select(nov, rel, static_mask, budget, nov_share=0.5):
    """两条通道各取一半名额，取并集，凑够 budget 个，按时间顺序返回下标。

    返回 (B, budget) 的 int32 下标，升序——升序即时间顺序（槽位编号 = 帧号*per_frame+格号），
    而 MemoryAttention 会按位置给 key 上 RoPE，所以顺序不能乱。

    并集通常不足 budget（两边有重叠）。空出来的名额**重复最后一个下标**补齐，
    与 GR00T 的 `_patch_union_mem_seq` 一致（`u[-1:].expand(budget - u.numel())`）。
    这样两边的记忆内容才等价，实验结果的差异才只能归因于选择规则本身。
    """
    B, N = nov.shape
    k_nov = max(1, int(round(budget * nov_share)))
    k_rel = max(0, budget - k_nov)

    rel = jnp.where(static_mask, rel, NEG)
    keep = _topk_mask(nov, k_nov, N)
    if k_rel > 0:
        keep = keep | _topk_mask(rel, k_rel, N)

    # 先按“是否入选”排序取前 budget，未入选的位置标成 N（哨兵），随后被最后一个真下标顶掉。
    order = jnp.where(keep, jnp.arange(N)[None, :], N)
    order = jnp.sort(order, axis=-1)[:, :budget]                 # 升序 = 时间顺序，哨兵沉底
    # 每行最后一个真下标：并集大小 - 1
    n_keep = jnp.sum(keep, axis=-1, keepdims=True)               # (B, 1)
    last = jnp.take_along_axis(order, jnp.maximum(n_keep - 1, 0), axis=-1)
    idx = jnp.where(order >= N, last, order)                     # 哨兵位 -> 重复最后一个
    return idx.astype(jnp.int32)


def gather_memory(mem_seq, mem_mask, idx):
    """按下标把候选池收缩成 (B, budget, d) 和 (B, budget)。"""
    return (jnp.take_along_axis(mem_seq, idx[..., None], axis=1),
            jnp.take_along_axis(mem_mask, idx, axis=1))


def selection_stats(nov, rel, static_mask, budget, nov_share=0.5):
    """诊断用：两条通道各自前 k 名的重合比例。

    这个数是整件事的生死线——如果两条通道选的高度重合，说明 relevance 没带来新信息，
    A2R 退化成 TokenDrop。训练时定期打出来看。
    """
    B, N = nov.shape
    k_nov = max(1, int(round(budget * nov_share)))
    k_rel = max(0, budget - k_nov)
    rel = jnp.where(static_mask, rel, NEG)
    a = _topk_mask(nov, k_nov, N)
    b = _topk_mask(rel, k_rel, N) if k_rel > 0 else jnp.zeros_like(a)
    inter = jnp.sum(a & b, axis=-1)
    union = jnp.sum(a | b, axis=-1)
    return dict(overlap=jnp.mean(inter / jnp.maximum(union, 1)),
                n_union=jnp.mean(union.astype(jnp.float32)))