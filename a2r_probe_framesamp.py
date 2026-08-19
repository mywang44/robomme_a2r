#!/usr/bin/env python3
"""A2R 前置探测：在 FrameSamp 记忆上，比较「模型注意力」和「novelty」各自挑出的 patch。

为什么用 FrameSamp 的 checkpoint：FrameSamp 的记忆就是「均匀 8 帧 × 每帧全部 64 个方块
= 512」，和 A2R 想要的无预筛候选池完全一致。所以模型看到的是它训练时天天见的分布，
不存在分布外的问题（换成 TokenDrop 的 checkpoint 就有）。

输出：
  * 每个任务一个 HTML：真实画面 + 8x8 网格，按「谁选中了这个格子」上色
      绿 = 只有注意力选   蓝 = 只有 novelty 选   黄 = 两个都选   暗 = 都没选
  * 一个 npz：18 层的注意力分数全存，改 --layer 重画不用再跑模型
  * 逐层统计表（集中度 / 熵 / 跨样本重合度），用来在自由池上复核该取哪一层

渲染部分改自 memory_for_vlas/scripts/vis_ls_patches.py（同一套配色和卡片布局）。

  uv run python a2r_probe_framesamp.py --task "container hiding the red cube" \
      --episodes 2 --anchors 8 --layer 17
  uv run python a2r_probe_framesamp.py --from-npz /tmp/a2r_probe/dump.npz --layer 13
"""
from __future__ import annotations

import argparse, base64, dataclasses, io, json, os, pathlib, pickle

import numpy as np
from PIL import Image

ROOT = "data/robomme_preprocessed_data"
FEAT = ROOT + "/features"
SIDE, PER_FRAME = 8, 64          # 8x8 网格，单视角
CELL = 32                        # 每格 32 像素（build_robomme_dataset.py 的常数）
GREEN, BLUE, YELLOW = (60, 200, 80), (70, 130, 240), (240, 210, 60)


# ----------------------------------------------------------------- 渲染（改自 senior）
def overlay(img, cells, alpha=0.45):
    """cells: {(row, col): color}；img: HxWx3 uint8。半透明填充 + 描边。"""
    im = img.astype(np.float32).copy()
    H, W = im.shape[:2]
    ch, cw = H / SIDE, W / SIDE
    for (r, c), col in cells.items():
        y0, y1 = int(r * ch), int((r + 1) * ch)
        x0, x1 = int(c * cw), int((c + 1) * cw)
        im[y0:y1, x0:x1] = (1 - alpha) * im[y0:y1, x0:x1] + alpha * np.array(col, np.float32)
        im[y0:y0 + 2, x0:x1] = col; im[y1 - 2:y1, x0:x1] = col
        im[y0:y1, x0:x0 + 2] = col; im[y0:y1, x1 - 2:x1] = col
    return im.astype(np.uint8)


def b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def video_of(E):
    import imageio.v3 as iio
    d = f"{FEAT}/episode_{E}"
    f = [x for x in os.listdir(d) if x.startswith("original_video_")][0]
    return iio.imread(f"{d}/{f}"), f[len("original_video_"):-4]


# ----------------------------------------------------------------- 打分
def novelty_scores(emb, n_frames, protect_frame0=True):
    """emb: (512, 2048) 记忆特征，槽位 f*64+p。
    novelty = 这一格减去上一张候选帧同位置那一格，逐维取绝对值再平均。
    第 0 张没有上一张，按 TokenDrop 的哨兵做法给无穷大（可关）。"""
    v = emb.reshape(n_frames, PER_FRAME, -1).astype(np.float32)
    prev = np.concatenate([v[:1], v[:-1]], axis=0)
    d = np.abs(v - prev).mean(axis=-1)                    # (n_frames, 64)
    if protect_frame0:
        d[0, :] = np.inf
    return d.reshape(-1)


def topk_set(scores, k, valid):
    s = np.where(valid, scores, -np.inf)
    return set(np.argsort(s)[::-1][:k].tolist())


# ----------------------------------------------------------------- 计算
def compute(args):
    import jax, jax.numpy as jnp
    import openpi.models.model as _model
    import mme_vla_suite.training.config as _config
    from openpi.training.data_loader import transform_dataset
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.shared.data_utils import even_sampling_indices
    from mme_vla_suite.training import dataset as _ds

    tmap = json.load(open("/tmp/a2r_task_map.json")) if os.path.exists("/tmp/a2r_task_map.json") else {}
    if not tmap:
        for d in os.listdir(FEAT):
            if d.startswith("episode_") and d.split("_", 1)[1].isdigit():
                try:
                    v = [f for f in os.listdir(f"{FEAT}/{d}") if f.startswith("original_video_")]
                except OSError:
                    continue
                if v:
                    tmap[d.split("_", 1)[1]] = v[0][len("original_video_"):-4]
        json.dump(tmap, open("/tmp/a2r_task_map.json", "w"))
    if args.list:
        for t in sorted(set(tmap.values())):
            print("  [%3d] %s" % (sum(1 for v in tmap.values() if v == t), t[:88]))
        return None

    eps = sorted(int(e) for e, t in tmap.items() if args.task.lower() in t.lower())
    print("匹配 %d 条 episode，用前 %d 条: %s" % (len(eps), args.episodes, eps[:args.episodes]))
    eps = eps[:args.episodes]
    if not eps:
        raise SystemExit("没有匹配的 episode")

    ck = pathlib.Path(args.ckpt)
    hcp = ck.parent / "history_config.txt"
    hc = hcp.read_text().strip() if hcp.exists() else "perceptual-framesamp-modul.yaml"
    print("history_config:", hc)
    hcfg = get_history_config(hc)
    assert hcfg.perceptual_memory.type == "frame_sampling", \
        "这个探测要用 FrameSamp 的 checkpoint（记忆才是无预筛的 8 帧 x 64），当前是 %s" \
        % hcfg.perceptual_memory.type

    tc = _config.get_config("mme_vla_suite")
    mc = dataclasses.replace(tc.model, history_config=hcfg, use_history=True)
    print("加载模型 ...", flush=True)
    model = mc.load(_model.restore_params(ck / "params", dtype=jnp.bfloat16))
    dcfg = tc.data.create(tc.assets_dirs, tc.model)

    ds = _ds.RoboMMEDataset(dataset_path=ROOT, data_config=dcfg,
                            history_config=hcfg, action_horizon=mc.action_horizon)
    N = len(ds)
    tds = transform_dataset(ds, dcfg, skip_norm_stats=False)
    n_frames = int(hcfg.budget) // (hcfg.token_per_image * hcfg.num_views)
    print("数据集 %d 个样本；记忆 = %d 帧 x %d 格" % (N, n_frames, hcfg.token_per_image))

    def ep_of(i):
        with open(f"{ROOT}/data/{i}.pkl", "rb") as f:
            return int(np.asarray(pickle.load(f)["epis_idx"]).ravel()[0])

    def find_range(E):
        if ep_of(0) > E or ep_of(N - 1) < E:
            return None
        a, b = 0, N - 1
        while a < b:
            m = (a + b) // 2
            if ep_of(m) < E: a = m + 1
            else: b = m
        if ep_of(a) != E:
            return None
        s, lo, hi = a, a, N - 1
        while lo < hi:
            m = (lo + hi + 1) // 2
            if ep_of(m) <= E: lo = m
            else: hi = m - 1
        return s, lo

    def batchify(x):
        if isinstance(x, dict):
            return {k: batchify(v) for k, v in x.items()}
        a = np.asarray(x)
        return jnp.asarray(a)[None, ...] if a.dtype.kind in "biufc" else x

    rows = []
    for E in eps:
        r = find_range(E)
        if r is None:
            print("  ep%d 定位失败（epis_idx 可能非单调）" % E); continue
        s, e = r
        idxs = list(range(s, e + 1, args.stride))[:args.anchors]
        print("  ep%d -> flat %d..%d，取 %d 个锚点" % (E, s, e, len(idxs)), flush=True)
        for i in idxs:
            item = tds[i]
            b = batchify(item)
            o = HistAugObservation.from_dict(b)
            out = model.compute_loss(jax.random.key(0), o, b["actions"],
                                     train=False, return_mem_rel=True)
            attn = np.asarray(jnp.asarray(out[2], dtype=jnp.float32))[:, 0, :]   # (18, 512)
            emb = np.asarray(item["static_image_emb"], np.float32)               # (512, 2048)
            mask = np.asarray(item["static_mask"], bool)                         # (512,)
            T = int(np.asarray(item.get("step_idx", -1)).ravel()[0]) if "step_idx" in item else -1
            rows.append(dict(ep=E, flat=i, attn=attn, emb=emb, mask=mask, step=T))
            print("    flat %-7d 有效 %d/%d" % (i, mask.sum(), mask.size), flush=True)

    if not rows:
        raise SystemExit("没采到样本")
    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "dump.npz")
    np.savez_compressed(
        npz,
        attn=np.stack([r["attn"] for r in rows]),
        emb=np.stack([r["emb"] for r in rows]).astype(np.float16),
        mask=np.stack([r["mask"] for r in rows]),
        ep=np.array([r["ep"] for r in rows]),
        flat=np.array([r["flat"] for r in rows]),
        step=np.array([r["step"] for r in rows]),
        n_frames=n_frames, task=args.task)
    print("已存", npz)
    return npz


# ----------------------------------------------------------------- 统计 + 出图
def render(npz_path, args):
    from mme_vla_suite.shared.data_utils import even_sampling_indices
    d = np.load(npz_path, allow_pickle=True)
    attn, emb, mask = d["attn"], d["emb"].astype(np.float32), d["mask"]
    eps, steps = d["ep"], d["step"]
    n_frames = int(d["n_frames"])
    task = str(d["task"])
    N, L, S = attn.shape
    K = args.topk

    # ---- 逐层统计：在自由池上复核该用哪一层 ----
    print()
    print("逐层统计（%d 个样本；均匀分布时 top%d 占比 = %.1f%%）"
          % (N, K, 100.0 * K / max(int(np.median(mask.sum(1))), 1)))
    print("  层    top%d占比      熵      跨样本重合度    与novelty重合" % K)
    stats = []
    for l in range(L):
        tops, share, ent, iou = [], [], [], []
        for i in range(N):
            a = attn[i, l]
            v = mask[i]
            nv = novelty_scores(emb[i], n_frames, not args.no_protect_frame0)
            A = topk_set(a, K, v)
            B = topk_set(nv, K, v)
            tops.append(A)
            share.append(sum(a[j] for j in A) / max(a[v].sum(), 1e-9))
            p = np.clip(a[v], 1e-9, None); p = p / p.sum()
            ent.append(float(-(p * np.log(p)).sum()))
            iou.append(len(A & B) / max(len(A | B), 1))
        ov = np.mean([len(tops[x] & tops[y]) / K
                      for x in range(min(N, 16)) for y in range(x + 1, min(N, 16))]) if N > 1 else 0.0
        stats.append((l, np.mean(share), np.mean(ent), ov, np.mean(iou)))
        print("  %2d     %5.1f%%     %.3f       %5.1f%%        %5.1f%%"
              % (l, np.mean(share) * 100, np.mean(ent), ov * 100, np.mean(iou) * 100))

    print()
    print("候选层（集中度高、跨样本重合度低）：")
    for l, sh, en, ov, iu in sorted(stats, key=lambda r: -(r[1] - r[3]))[:4]:
        print("   层 %2d: top%d %.1f%%, 熵 %.3f, 跨样本重合 %.1f%%, 与novelty重合 %.1f%%"
              % (l, K, sh * 100, en, ov * 100, iu * 100))

    # ---- HTML ----
    os.makedirs(args.out, exist_ok=True)
    cards, ious = [], []
    vid_cache = {}
    for i in range(N):
        E, T = int(eps[i]), int(steps[i])
        if E not in vid_cache:
            vid_cache[E] = video_of(E)
        vid, instr = vid_cache[E]
        a, v = attn[i, args.layer], mask[i]
        nv = novelty_scores(emb[i], n_frames, not args.no_protect_frame0)
        A, B = topk_set(a, K, v), topk_set(nv, K, v)
        iou = len(A & B) / max(len(A | B), 1)
        ious.append(iou)
        # 记忆用的是哪几帧：FrameSamp 的均匀采样，给定锚点是确定的
        fr = even_sampling_indices(T, n_frames) if T >= 0 else list(range(n_frames))
        if len(fr) < n_frames:
            fr = fr + [fr[-1]] * (n_frames - len(fr))
        pics = []
        for f in range(n_frames):
            cells = {}
            for p in range(PER_FRAME):
                j = f * PER_FRAME + p
                if not v[j]:
                    continue
                col = (YELLOW if (j in A and j in B) else
                       GREEN if j in A else BLUE if j in B else None)
                if col is not None:
                    cells[(p // SIDE, p % SIDE)] = col
            src = int(fr[f]) if f < len(fr) else 0
            img = vid[min(src, len(vid) - 1)][:, :SIDE * CELL]      # 左半边 = 记忆用的那个视角
            na = sum(1 for j in A if j // PER_FRAME == f)
            nb = sum(1 for j in B if j // PER_FRAME == f)
            pics.append("<figure><img src='data:image/jpeg;base64,%s'>"
                        "<figcaption>帧 %d（-%d）　注意力 %d ／ novelty %d</figcaption></figure>"
                        % (b64(overlay(img, cells)), src, max(T - src, 0), na, nb))
        cards.append("<div class='card'><h4>ep%d　锚点 t=%d　有效 %d/%d　iou=%.3f</h4>%s</div>"
                     % (E, T, v.sum(), v.size, iou, "".join(pics)))

    html = """<html><head><meta charset='utf-8'><style>
body{font-family:sans-serif;background:#111;color:#eee;margin:20px}
.card{background:#1a1a1a;padding:10px;border-radius:8px;margin:14px 0}
figure{display:inline-block;margin:4px;text-align:center}
figure img{width:190px;display:block;border-radius:3px}
figcaption{font-size:11px;color:#999;margin-top:2px}
h4{margin:4px 0 8px;font-size:13px;color:#ccc}
.legend span{padding:3px 12px;margin-right:10px;border-radius:4px;font-size:13px}
</style></head><body>
<h2>A2R 前置探测 — FrameSamp 记忆（均匀 %d 帧 × 每帧全部 %d 格，无预筛）</h2>
<p style='color:#888'>任务筛选：%s　｜　第 %d 层注意力　｜　各取前 %d　｜　平均 iou = %.3f</p>
<p class='legend'><span style='background:rgb%s;color:#000'>只有注意力选</span>
<span style='background:rgb%s;color:#000'>只有 novelty 选</span>
<span style='background:rgb%s;color:#000'>两个都选</span>
<span style='color:#666'>暗淡 = 都没选</span></p>
%s</body></html>""" % (n_frames, PER_FRAME, task or "(全部)", args.layer, K,
                       float(np.mean(ious)), str(GREEN), str(BLUE), str(YELLOW), "".join(cards))
    p = os.path.join(args.out, "a2r_probe_L%02d.html" % args.layer)
    open(p, "w", encoding="utf-8").write(html)
    print("\n已写出", p, " 平均 iou = %.3f" % float(np.mean(ious)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ckpts/mme_vla_suite/perceptual-framesamp-modul_repro/70000")
    ap.add_argument("--task", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--no-protect-frame0", action="store_true")
    ap.add_argument("--out", default="/tmp/a2r_probe")
    ap.add_argument("--from-npz", default="")
    args = ap.parse_args()
    npz = args.from_npz or compute(args)
    if npz:
        render(npz, args)


if __name__ == "__main__":
    main()