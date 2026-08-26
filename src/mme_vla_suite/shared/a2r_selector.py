"""A2R 推理端的滚动记忆选择器。

逐行移植自 memory_for_vlas 的 `gr00t/model/modules/fs_patch_union.py::PatchUnionSelector`，
行为对齐参考实现：

  * 两个独立的堆，跨整条 episode 累积（每项连 img/pos/state 三个向量一起存——
    pos_emb 是按**绝对帧号**索引的、编码了时间，state 也是每帧一个，只存 img 复原不出来）：
      diff_heap  —— novelty（token 空间，当前帧减去**上一个被打分的帧**）
      attn_heap  —— relevance（动作对当前帧格子的注意力）
  * `read()` 把两个堆合并、按 (帧号, 格号) 去重、按时间排序、取前 budget；
    不足 budget 用**当前帧的格子**补齐。
  * `observe()` 是「后写」：relevance **每次调用**都写，novelty **每 stride 次**才写。
  * 第 0 帧的格子以 1e9 无条件入 diff_heap（TokenDrop 的哨兵）。

与参考实现唯一的偏差，已在 `observe` 里注明：GR00T 是双视角，哨兵只保护「前视角那一半」；
RoboMME 这边 num_views=1，所谓「前视角」就是全部 64 个格子，所以按 n_img // num_views 算。
"""
from __future__ import annotations

import heapq

import numpy as np


class PatchUnionSelector:
    def __init__(self, budget: int = 512, diff_share: float = 0.5, stride: int = 1,
                 num_views: int = 1):
        self.budget = int(budget)
        self.n_diff = max(1, int(round(budget * diff_share)))
        self.n_attn = max(0, self.budget - self.n_diff)
        self.stride = max(1, int(stride))
        self.num_views = max(1, int(num_views))
        self.reset()

    def reset(self):
        self.diff_heap: list = []   # (score, step, idx, token)
        self.attn_heap: list = []
        self.last_tok: np.ndarray | None = None   # 上一个被打分的帧的 (n_img, d)
        self.step = -1

    @staticmethod
    def _push(heap, cap, item):
        heapq.heappush(heap, item)
        if len(heap) > cap:
            heapq.heappop(heap)          # 最小堆，弹掉分数最低的

    def observe(self, vis, pos, state, attn=None) -> None:
        """写：把当前帧的格子推进两个堆。

        vis   : (n_img, d1) 当前帧的视觉 token
        pos   : (n_img, d2) 对应的位置嵌入（含时间，按绝对帧号取的）
        state : (n_img, d3) 对应的状态嵌入（整帧同一个，已按格子复制）
        attn  : (n_img,) 动作对这些格子的注意力；第一次调用时为 None（还没跑过前向）
        """
        self.step += 1
        vis = np.asarray(vis); pos = np.asarray(pos); state = np.asarray(state)
        n_img = vis.shape[0]
        pack = lambda i: (vis[i], pos[i], state[i])

        if self.step == 0:
            # TokenDrop 的哨兵：第 0 帧无条件入堆。GR00T 是双视角、只保护「前视角那一半」；
            # 这里 num_views=1，所谓前视角就是全部格子，所以按 n_img // num_views 算。
            per_view = max(1, n_img // self.num_views)
            for i in range(min(per_view, n_img)):
                self._push(self.diff_heap, self.n_diff, (1e9, 0, i, pack(i)))
            self.last_tok = vis.astype(np.float32)
            return

        if self.step % self.stride == 0:
            d = np.abs(vis.astype(np.float32) - self.last_tok).mean(axis=-1)
            self.last_tok = vis.astype(np.float32)
            for i in range(n_img):
                sc = float(d[i])
                if sc > 1e-4:
                    self._push(self.diff_heap, self.n_diff, (sc, self.step, i, pack(i)))

        if attn is not None and self.n_attn > 0:
            attn = np.asarray(attn)
            for i in range(n_img):
                self._push(self.attn_heap, self.n_attn,
                           (float(attn[i]), self.step, i, pack(i)))

    def read(self, vis_current, pos_current, state_current):
        """读：合并两个堆 -> (budget, ·) 的 img / pos / state 三份 + (budget,) mask。

        本次决策用的记忆只含 `history < t`；当前帧只在堆没填满时作为补齐用。
        """
        vis_current = np.asarray(vis_current)
        pos_current = np.asarray(pos_current)
        state_current = np.asarray(state_current)

        seen, entries = set(), []
        for heap in (self.diff_heap, self.attn_heap):
            for (_s, t, i, pk) in heap:
                if (t, i) not in seen:
                    seen.add((t, i))
                    entries.append((t, i, pk))
        entries.sort(key=lambda e: (e[0], e[1]))        # 按 (帧号, 格号) 升序 = 时间顺序
        kept = entries[: self.budget]

        img = [p[0] for (_t, _i, p) in kept]
        pos = [p[1] for (_t, _i, p) in kept]
        sta = [p[2] for (_t, _i, p) in kept]
        n_img = vis_current.shape[0]
        k = 0
        while len(img) < self.budget:                   # 补齐：用当前帧的格子
            j = k % n_img
            img.append(vis_current[j]); pos.append(pos_current[j]); sta.append(state_current[j])
            k += 1
        return (np.stack(img, 0), np.stack(pos, 0), np.stack(sta, 0),
                np.ones((self.budget,), dtype=bool))

    def stats(self) -> dict:
        """诊断：堆的占用和覆盖的帧数。评测时打出来确认记忆确实在长。"""
        steps = {t for h in (self.diff_heap, self.attn_heap) for (_s, t, _i, _p) in h}
        return dict(step=self.step, n_diff=len(self.diff_heap),
                    n_attn=len(self.attn_heap), n_frames=len(steps))
