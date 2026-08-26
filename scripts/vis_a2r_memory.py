"""把 A2R 的两条打分通道分别画在真实图片上（训练集，训练时怎么取帧就怎么取）。

产出两张**分开的**图：
  <out>/sample<idx>_relevance.png  —— 动作 -> 记忆 的注意力（pass A 抓的那一趟）
  <out>/sample<idx>_novelty.png    —— 相邻候选帧同位置格子的像素级特征差
另存 <out>/sample<idx>_scores.npz，想换配色重画不用再跑模型。

用法（在 pod 上，仓库根目录）：
  unset PYTHONPATH
  export OPENPI_DATA_HOME=$WS/.cache/openpi
  uv run python scripts/vis_a2r_memory.py \
      --ckpt runs/ckpts/mme_vla_suite/a2r_union/79999 \
      --history-config perceptual-a2r-modul.yaml \
      --find unmask --out vis_a2r

设计上刻意**不重新实现任何打分逻辑**：候选帧的选法直接调 dataset/mem_buffer 的函数，
两条分数直接从模型 forward 里劫持 a2r_select.union_select 的入参拿。
仓库里那两处代码怎么改，这里就跟着变，不会出现「脚本和训练不一致」。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys

import numpy as np


# ---------------------------------------------------------------- 参数

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="checkpoint 目录，例如 runs/ckpts/mme_vla_suite/a2r_union/79999")
    p.add_argument("--config", default="mme_vla_suite", help="TrainConfig 名字")
    p.add_argument("--history-config", default=None,
                   help="history yaml 文件名；不给就读 ckpt 上一级的 history_config.txt")
    p.add_argument("--dataset-path", default="data/robomme_preprocessed_data",
                   help="训练用的是这个；注册配置里的默认值 data/robomme 是错的")
    p.add_argument("--raw-h5", default="data/robomme_data_h5", help="原始 h5 目录（取真实图片）")
    p.add_argument("--sample-idx", default="", help="逗号分隔的样本下标；留空则用 --find 随机找")
    p.add_argument("--find", default="", help="按 prompt 子串随机找样本，例如 unmask")
    p.add_argument("--find-tries", type=int, default=400)
    p.add_argument("--min-step", type=int, default=120,
                   help="只要 step_idx 够大的样本，历史才够长、候选帧才铺得开")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--out", default="vis_a2r")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=130)
    return p.parse_args()


# ---------------------------------------------------------------- h5 帧映射

def build_episode_map(raw_dir, sort_files):
    import h5py
    names = os.listdir(raw_dir)
    names = sorted(names) if sort_files else names
    names = [f for f in names if f.endswith(".h5")]
    mapping, g = {}, 0
    for fname in names:
        with h5py.File(os.path.join(raw_dir, fname), "r") as d:
            eps = sorted(int(k.split("_")[1]) for k in d.keys() if k.startswith("episode_"))
        for e in eps:
            mapping[g] = (fname, e)
            g += 1
    return mapping


def read_h5_frames(raw_dir, fname, ep, steps):
    import h5py
    out = {}
    with h5py.File(os.path.join(raw_dir, fname), "r") as d:
        grp = d[f"episode_{ep}"]
        for t in steps:
            key = f"timestep_{t}"
            out[t] = np.asarray(grp[key]["obs"]["front_rgb"][()]) if key in grp else None
    return out


def resolve_episode_map(raw_dir, epis_idx, anchor_step, anchor_image):
    """两种 listdir 顺序各试一次，用锚点帧原图逐像素校验。"""
    for sort_files in (False, True):
        try:
            m = build_episode_map(raw_dir, sort_files)
        except Exception as e:                                   # noqa: BLE001
            print(f"[h5] 建映射失败 (sorted={sort_files}): {e}")
            continue
        if epis_idx not in m:
            continue
        fname, ep = m[epis_idx]
        got = read_h5_frames(raw_dir, fname, ep, [anchor_step]).get(anchor_step)
        if got is not None and got.shape == anchor_image.shape and np.array_equal(got, anchor_image):
            print(f"[h5] 映射校验通过 (sorted={sort_files}): "
                  f"epis_idx {epis_idx} -> {fname}::episode_{ep}")
            return m
        print(f"[h5] 映射校验失败 (sorted={sort_files}): {fname}::episode_{ep} 的第 "
              f"{anchor_step} 帧和 pkl 里的原图对不上")
    return None


# ---------------------------------------------------------------- 画图

def pad_resize_224(img_uint8):
    """和 mem_buffer.add_buffer 里完全一样的预处理，保证 8x8 格子对得上像素。"""
    import jax.numpy as jnp
    from openpi.shared import image_tools
    x = jnp.asarray(img_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0)[None]
    x = image_tools.resize_with_pad(x, 224, 224)[0]
    return np.clip((np.asarray(x, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)


def draw_panel_grid(frames, images, heat, valid, title, subtitle, out_path,
                    cmap, dpi, sentinel_frame=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    scale = heat.copy()
    if sentinel_frame is not None and n > 1:
        scale = np.delete(scale, sentinel_frame, axis=0)
        vmask = np.delete(valid, sentinel_frame, axis=0)
    else:
        vmask = valid
    fin = scale[vmask] if vmask.any() else scale.reshape(-1)
    vmin, vmax = float(np.min(fin)), float(np.max(fin))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.75))
    axes = np.atleast_1d(axes).reshape(-1)
    im = None
    for i in range(rows * cols):
        ax = axes[i]
        ax.set_xticks([]); ax.set_yticks([])
        if i >= n:
            ax.axis("off"); continue
        bg = images[i]
        if bg is None:
            # 只可能是右侧补齐出来的占位帧（frames[i] < 0），真实帧缺图在上游就已经退出了
            ax.axis("off")
            ax.set_title("(padded)", fontsize=7, color="gray")
            continue
        ax.imshow(bg, interpolation="bilinear")
        if not valid[i].any():
            ax.set_title(f"t={frames[i]}  (padded)", fontsize=7, color="gray")
            continue
        if sentinel_frame is not None and i == sentinel_frame:
            ax.imshow(np.ones_like(heat[i]), cmap=cmap, vmin=0.0, vmax=1.0,
                      alpha=0.55, extent=(0, 224, 224, 0), interpolation="nearest")
            ax.set_title(f"t={frames[i]}  SENTINEL (forced in)", fontsize=7, color="crimson")
            continue
        im = ax.imshow(heat[i], cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.55,
                       extent=(0, 224, 224, 0), interpolation="nearest")
        share = float(np.sum(np.clip(heat[i] - vmin, 0, None)))
        ax.set_title(f"t={frames[i]}   sum={share:.3g}", fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    if im is not None:
        cax = fig.add_axes([0.25, 0.015, 0.5, 0.014])
        fig.colorbar(im, cax=cax, orientation="horizontal")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[out] {out_path}")



def _batchify(item, jnp):
    """给每个叶子加一个 batch 维；嵌套字典要递归（image 是按相机分的 dict），
    字符串/object 数组直接丢掉（prompt 之类，模型用的是 tokenized_prompt）。

    注意不能按 dtype.kind 简单过滤：bfloat16 在 numpy 侧的 kind 是 'V'，
    当成"非数值"跳过的话 static_image_emb 就没了，后面报的错会离现场十万八千里。
    """
    dropped = set()

    def rec(v, path):
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                r = rec(x, f"{path}.{k}" if path else k)
                if r is not None:
                    out[k] = r
            return out or None
        if isinstance(v, (str, bytes)):
            dropped.add(path)
            return None
        a = np.asarray(v)
        if a.dtype.kind in ("O", "U", "S"):
            dropped.add(path)
            return None
        return jnp.asarray(a)[None]

    batch = {}
    for k, v in item.items():
        r = rec(v, k)
        if r is not None:
            batch[k] = r
    return batch, dropped


# ---------------------------------------------------------------- 主流程

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    import dataclasses

    import jax
    import jax.numpy as jnp
    # 注意：mme_vla_suite 这个 config 注册在自己的模块里，不在 openpi.training.config
    import mme_vla_suite.training.config as _config
    import openpi.training.checkpoints as _checkpoints
    from openpi.models import model as _model
    from openpi.training.data_loader import transform_dataset
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.dataset import RoboMMEDataset
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.models.integration import a2r_select as _a2r

    ckpt = pathlib.Path(args.ckpt)
    cfg = _config.get_config(args.config)

    hc_name = args.history_config
    if hc_name is None:
        for cand in (ckpt / "history_config.txt", ckpt.parent / "history_config.txt"):
            if cand.exists():
                hc_name = cand.read_text().strip()
                print(f"[cfg] history_config 取自 {cand}: {hc_name}")
                break
    if hc_name is None:
        sys.exit("找不到 history_config，请用 --history-config 指定 yaml 文件名")

    history_config = get_history_config(hc_name)
    pm = history_config.perceptual_memory
    if pm.type != "a2r":
        sys.exit(f"这个 history_config 的 perceptual_memory.type = {pm.type}，不是 a2r")

    n_frames = int(getattr(pm, "cand_frames", 8))
    tpi = int(history_config.token_per_image)
    n_views = int(history_config.num_views)
    per_frame = tpi * n_views
    side = int(round(np.sqrt(tpi)))
    layer = int(getattr(pm, "attn_layer", 17))
    budget = int(history_config.budget)
    assert side * side == tpi, f"token_per_image={tpi} 不是完全平方数，画不成方格"
    print(f"[cfg] 候选 {n_frames} 帧 x {per_frame} 格 = {n_frames * per_frame}，"
          f"budget {budget}，grid {side}x{side}，views {n_views}，attn layer {layer}")

    # 注册的 TrainConfig 里 history_config=None（训练时由命令行覆盖）。必须在这里补上，
    # 否则 model.create/load 出来的是个没有记忆模块的 pi05，权重对不上、也没有 _a2r_pick。
    model_cfg = dataclasses.replace(cfg.model, history_config=hc_name, use_history=True)

    dataset_path = args.dataset_path
    data_config = cfg.data.create(cfg.assets_dirs, model_cfg)
    if data_config.norm_stats is None and data_config.asset_id is not None:
        # 和 create_trained_policy 一样，优先用 checkpoint 里带的归一化统计量
        data_config = dataclasses.replace(
            data_config,
            norm_stats=_checkpoints.load_norm_stats(ckpt / "assets", data_config.asset_id))
        print("[cfg] norm_stats 取自 checkpoint/assets")
    ds = RoboMMEDataset(dataset_path=dataset_path, data_config=data_config,
                        history_config=history_config,
                        action_horizon=cfg.model.action_horizon)
    ds_t = transform_dataset(ds, data_config, skip_norm_stats=False)
    print(f"[data] {dataset_path}  共 {len(ds.dataset)} 个样本")

    # ---- 选样本
    if args.sample_idx:
        picks = [int(s) for s in args.sample_idx.split(",")]
    else:
        picks = []
        total = len(ds.dataset)
        seen = {}
        n_deep = 0
        for _ in range(args.find_tries):
            j = random.randrange(total)
            raw = ds.dataset[j]
            pr = str(raw["prompt"])
            seen[pr] = seen.get(pr, 0) + 1
            if int(raw["step_idx"].item()) < args.min_step:
                continue
            n_deep += 1
            if args.find and args.find.lower() not in pr.lower():
                continue
            picks.append(j)
            if len(picks) >= args.n_samples:
                break
        if not picks:
            print(f"\n随机试了 {args.find_tries} 个样本，其中 step_idx>={args.min_step} 的有 "
                  f"{n_deep} 个，没有一个 prompt 含 '{args.find}'。")
            print("撞到的 prompt（按出现次数排序；prompt 存的是 task_goal，不是任务名）：")
            for pr, c in sorted(seen.items(), key=lambda kv: -kv[1])[:25]:
                print(f"  {c:3d}x  {pr}")
            sys.exit(1)
    print(f"[data] 选中样本 {picks}")

    # ---- 载模型
    print("[model] 载入权重…")
    model = model_cfg.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))

    grabbed = {}
    orig_union = _a2r.union_select

    def spy(nov, rel, static_mask, bud, nov_share=0.5):
        grabbed["nov"] = np.asarray(jax.device_get(nov), dtype=np.float64)
        grabbed["rel"] = np.asarray(jax.device_get(rel), dtype=np.float64)
        grabbed["mask"] = np.asarray(jax.device_get(static_mask))
        return orig_union(nov, rel, static_mask, bud, nov_share)

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    epis_map = None

    for idx in picks:
        raw = ds.dataset[idx]
        epis_idx = int(raw["epis_idx"].item())
        step_idx = int(raw["step_idx"].item())
        prompt = str(raw["prompt"])
        exec_start = int(raw["exec_start_idx"].item())
        print(f"\n=== sample {idx}: episode {epis_idx}, step {step_idx}, "
              f"exec_start {exec_start}\n    prompt: {prompt}")

        # 候选帧号不去猜函数名（不同版本的 mem_buffer 走 prepare_a2r_pool 还是
        # prepare_frame_sampling 不一定），只劫持两条路都必经的加载口子：
        # _gather_history_feat 的第一个参数就是这一次要读哪些帧。
        seen_frames = {}
        orig_gather = ds._gather_history_feat

        def gather_spy(indices_to_load, *a, **kw):
            seen_frames["v"] = [int(t) for t in indices_to_load]
            return orig_gather(indices_to_load, *a, **kw)

        ds._gather_history_feat = gather_spy
        try:
            item = ds_t[idx]
        finally:
            ds._gather_history_feat = orig_gather
        if "v" not in seen_frames:
            sys.exit("没抓到候选帧号——这个样本没走感知记忆的加载路径")
        frames = seen_frames["v"]
        print(f"    候选帧 ({len(frames)}): {frames}")

        batch, dropped = _batchify(item, jnp)
        if dropped:
            print(f"    (非数值字段已跳过: {', '.join(sorted(dropped))})")
        obs = HistAugObservation.from_dict(batch)
        actions = batch["actions"]

        grabbed.clear()
        _a2r.union_select = spy
        try:
            model.compute_loss(jax.random.key(0), obs, actions, train=False)
        finally:
            _a2r.union_select = orig_union
        if "rel" not in grabbed:
            sys.exit("没抓到 union_select 的入参——这个 ckpt 的 history_config 可能不是 a2r")

        # 帧数以实际抓到的候选池长度为准，不用配置里的值硬套
        n_cand = int(grabbed["nov"].shape[-1])
        n_f = n_cand // per_frame
        if n_f * per_frame != n_cand:
            sys.exit(f"候选池长度 {n_cand} 不是每帧格数 {per_frame} 的整数倍，"
                     f"槽位布局和预期不符，拒绝画图")
        if len(frames) < n_f:
            # 历史不够长时右侧是补齐帧，mask 已经是 False，这里只补个占位帧号
            frames = frames + [-1] * (n_f - len(frames))
        frames = frames[:n_f]
        nov = grabbed["nov"][0][:n_cand].reshape(n_f, n_views, side, side)
        rel = grabbed["rel"][0][:n_cand].reshape(n_f, n_views, side, side)
        msk = grabbed["mask"][0][:n_cand].reshape(n_f, n_views, side, side)

        # ---- 真实图片
        images = [None] * n_f
        if not os.path.isdir(args.raw_h5):
            sys.exit(f"找不到原始 h5 目录 {args.raw_h5}，无法取真实图片背景。\n"
                     f"这个脚本不做灰底降级——没有真图就不出图。\n"
                     f"请用 --raw-h5 指向建这份特征时用的那批 h5。")
        if True:
            if epis_map is None:
                epis_map = resolve_episode_map(args.raw_h5, epis_idx, step_idx,
                                               np.asarray(raw["image"]))
                if epis_map is None:
                    sys.exit("h5 帧映射两种顺序都校验失败，拒绝画可能张冠李戴的图。"
                             "请确认 --raw-h5 指向的是建这份特征时用的那批 h5。")
            fname, ep = epis_map[epis_idx]
            got = read_h5_frames(args.raw_h5, fname, ep, [t for t in frames if t >= 0])
            for i, t in enumerate(frames):
                if t >= 0 and got.get(t) is not None:
                    images[i] = pad_resize_224(got[t])
            missing = [frames[i] for i in range(n_f)
                       if images[i] is None and frames[i] >= 0]
            if missing:
                sys.exit(f"这些候选帧在 h5 里没找到: {missing}\n"
                         f"episode {epis_idx} -> {fname}::episode_{ep}，"
                         f"帧号或映射有问题，拒绝出图。")

        tag = f"ep{epis_idx}_step{step_idx}_sample{idx}"
        np.savez(out_dir / f"{tag}_scores.npz", nov=nov, rel=rel, mask=msk,
                 frames=np.asarray(frames), step_idx=step_idx, epis_idx=epis_idx,
                 exec_start_idx=exec_start, prompt=prompt)

        for v in range(n_views):
            vsuf = "" if n_views == 1 else f"_view{v}"
            sub = (f"episode {epis_idx}, anchor step {step_idx} (exec starts at {exec_start})"
                   f" | {prompt[:78]}")
            draw_panel_grid(
                frames, images, rel[:, v], msk[:, v],
                f"RELEVANCE  -  action->memory attention, layer {layer}",
                sub, out_dir / f"{tag}{vsuf}_relevance.png", "inferno", args.dpi)
            draw_panel_grid(
                frames, images, nov[:, v], msk[:, v],
                "NOVELTY  -  feature diff vs previous candidate frame",
                sub, out_dir / f"{tag}{vsuf}_novelty.png", "viridis", args.dpi,
                sentinel_frame=0)

        vm = msk[:, 0].reshape(n_f, -1).any(axis=1)
        rv = rel[:, 0].reshape(n_f, -1)
        tot = max(rv[vm].sum(), 1e-9)
        print("    每帧 relevance 占比: " + "  ".join(
            f"t{frames[i]}:{rv[i].sum() / tot * 100:4.1f}%"
            for i in range(n_f) if vm[i]))
        n_demo = sum(1 for t in frames if 0 <= t < exec_start)
        demo_share = sum(rv[i].sum() for i in range(n_f)
                         if vm[i] and 0 <= frames[i] < exec_start) / tot
        print(f"    候选帧里看视频阶段 (t<{exec_start}) 有 {n_demo}/{n_f} 帧，"
              f"吃掉 {demo_share * 100:.1f}% 的 relevance")


if __name__ == "__main__":
    main()
