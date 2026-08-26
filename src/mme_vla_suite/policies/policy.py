from collections.abc import Sequence
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.shared.a2r_selector import PatchUnionSelector
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent

class MME_VLA_Policy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)
        
        
        self.config = model.history_config
        self.mem_buffer = None
        
        self.state_norm_stats = norm_stats['state']
        self.use_quantiles = use_quantiles
        
        self.reset()
        
    
    def _prepare_mem_buffer(self):
        if self.config is None or self.config.representation_type == "symbolic":
            self.mem_buffer = None
        elif self.config.representation_type == "recurrent":
            self.mem_buffer = MemoryBufferRecurrent(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                input_obs_horizon=self.config.streaming_obs_horizon,
                max_recur_steps=self.config.recurrent_memory.max_recur_steps,
                max_video_steps=self.config.recurrent_memory.max_pretraj_steps,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )
        else:
            self.mem_buffer = MemoryBuffer(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                compute_token_drop_score = self.config.perceptual_memory.type == "token_dropping",
                token_drop_stride=self.config.streaming_obs_horizon // 2,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )
        # A2R：跨整条 episode 累积的滚动记忆（novelty 堆 + relevance 堆），
        # 与 GR00T 推理端的 PatchUnionSelector 行为一致。随 mem_buffer 一起重建。
        self.a2r_selector = None
        if (self.config is not None
                and getattr(self.config, "perceptual_memory", None) is not None
                and self.config.perceptual_memory.type == "a2r"):
            pm = self.config.perceptual_memory
            self.a2r_selector = PatchUnionSelector(
                budget=int(self.config.budget),
                diff_share=float(getattr(pm, "nov_share", 0.5)),
                # stride 的单位是**策略调用次数**，不是环境步。策略每 streaming_obs_horizon
                # (=16) 个环境步才调用一次，所以相邻两次 observe 本身就隔了 16 个环境步。
                # 训练侧的候选帧是 linspace(0, anchor-1, 16)，相邻帧间隔 anchor/16——中位
                # episode 在中段约 14 步，和这 16 步基本一致。所以这里必须是 1：每次调用都
                # 算 novelty。（照搬 mem_buffer 的 token_drop_stride=8 是错的，那个 8 的单位
                # 是环境步；当成调用次数会变成每 128 个环境步才比一次，和训练差 16 倍。）
                stride=1,
                num_views=int(self.config.num_views),
            )

    @override
    def infer(self, obs: dict) -> dict:
        if self.config is not None and self.config.representation_type != "symbolic":
            assert len(self.mem_buffer._history_feats) > 0, \
                "history feats is empty, add buffer first"
                                        
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        self._rng, sample_rng = jax.random.split(self._rng)
    
        start_time = time.monotonic()
        _acts = self._sample_actions(sample_rng, observation, **self._sample_kwargs)
        if self.a2r_selector is not None:
            # A2R：sample_actions 额外带回「动作对 prefix 每一列的注意力」（跨去噪步累加）。
            _acts, _prefix_attn = _acts
        outputs = {"state": observation.state, "actions": _acts}
        model_time = time.monotonic() - start_time
        if self.a2r_selector is not None:
            # 「后写」：把当前帧的格子连同它们的注意力分数推进堆，供**下一次**决策使用。
            self._a2r_commit(np.asarray(_prefix_attn[0]))
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)      
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000
        
        return outputs
    
    @override
    def reset(self) -> None:
        del self.mem_buffer
        self._prepare_mem_buffer()
        self.step_idx = -1  
        self.exec_start_idx = 0
        self._rng = jax.random.key(self._seed)
            
    
    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        if obs.get("exec_start_idx", 0) > 0: # has video
            self.exec_start_idx = obs["exec_start_idx"]
        
        step_idx_list = list(range(self.step_idx+1, self.step_idx + len(images) + 1))
        self.mem_buffer.add_buffer(images, states, step_idx_list)
        self.step_idx += len(images)

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _a2r_current_frame(self):
        """当前这一帧的 (img, pos, state)，都摊平成 (num_views*64, ·)。"""
        f = self.mem_buffer._history_feats[self.step_idx]
        img = np.asarray(f["image_emb_8x8"]).reshape(-1, self.config.memory_feature.img.input_dim)
        pos = np.asarray(f["pos_emb_8x8"]).reshape(-1, self.config.memory_feature.pos.input_dim)
        state = np.repeat(np.asarray(f["state_emb"])[None, :], img.shape[0], axis=0)
        return img, pos, state

    def _a2r_commit(self, prefix_attn):
        """把 prefix 上的注意力折算成「每个记忆格子一个分数」，然后写进堆。

        两个必须成立的前提（都核对过，写在这里是为了以后改动时能立刻发现）：

        1. **粒度**：prefix 的图像 token 是 SigLIP 的 16x16=256 个/视角，而记忆里的格子是
           8x8=64 个/视角——后者由 `pool_tokens_to_size` 对 16x16 做 **2x2 平均池化**得到。
           所以这里按同样的 2x2 分块平均，空间对应是精确的，不是近似。

        2. **取哪几列**：记忆只用 base 相机（`eval.py` 的 `pack_buffer` 只传 `image_buffer`，
           不传 `wrist_image_buffer`），而 prefix 里有 base + wrist 两张图，`base_0_rgb` 排在
           前面（`robomme_policy.py` 的插入顺序 + `IMAGE_KEYS`）。所以取前 num_views*256 列
           正好是 base 那一张。**如果哪天图像顺序变了，这里会静默读错相机**——下面的断言只
           能挡住长度不够的情况，挡不住顺序颠倒，改 `robomme_policy.py` 的图像字典时请回来看。
        """
        v = int(self.config.num_views)
        per_view_tok = 256                      # SigLIP So400m/14 @224 -> 16x16
        side_in, side_out = 16, 8
        prefix_attn = np.asarray(prefix_attn, np.float32)
        assert prefix_attn.shape[0] >= v * per_view_tok, (
            f"prefix 注意力只有 {prefix_attn.shape[0]} 列，不够 {v} 个视角 x {per_view_tok}")
        a = prefix_attn[: v * per_view_tok]
        a = a.reshape(v, side_in, side_in)
        a = a.reshape(v, side_out, 2, side_out, 2).mean(axis=(2, 4))   # 2x2 平均池化
        scores = a.reshape(-1)                  # (v*64,)
        img, pos, state = self._a2r_current_frame()
        self.a2r_selector.observe(img, pos, state, scores)

    def _prepare_history(self, inputs: dict) -> dict:
        if self.config is None or self.config.representation_type == "symbolic":
            return inputs
        
        if self.config.representation_type == "recurrent":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = \
                self.mem_buffer.prepare_token_recurrent(
                    self.step_idx, self.exec_start_idx, history_feats_gather_fn)
            inputs["recur_image_emb"] = recur_image_emb
            inputs["recur_pos_emb"] = recur_pos_emb
            inputs["recur_state_emb"] = self._normalize_state(recur_state_emb)
            inputs["recur_mask"] = recur_mask
        elif self.config.representation_type == "perceptual":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            token_budget = self.config.budget
            
            if self.config.perceptual_memory.type == "token_dropping":
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_token_dropping(
                        self.step_idx, token_budget, history_feats_gather_fn)
            elif self.config.perceptual_memory.type == "a2r":
                # 「先读」：从两个滚动堆里合并出 budget 个记忆。本次决策只用 history<t，
                # 当前帧要等动作出来之后才由 infer() 调 observe() 写回（后写）。
                cur = self._a2r_current_frame()
                (static_image_emb, static_pos_emb,
                 static_state_emb, static_mask) = self.a2r_selector.read(*cur)
            else:
                token_per_image = self.config.token_per_image
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_frame_sampling(
                        self.step_idx, token_budget, token_per_image, history_feats_gather_fn)
            
            inputs["static_image_emb"] = static_image_emb
            inputs["static_pos_emb"] = static_pos_emb
            inputs["static_state_emb"] = self._normalize_state(static_state_emb)
            inputs["static_mask"] = static_mask
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")
        
    
        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata