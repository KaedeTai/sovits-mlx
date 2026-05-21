"""r6 — Continue r5 training on MLX with c_kl=1.0 restored.

Resumes from r5 e16 (loaded via convert_full.py) and runs 5 more epochs
(e17→e21) of the same v2ProTw fine-tune with the upstream-default KL loss
that r5 had to disable as a workaround for an MPS view-stride bug.

Pragmatic design choices (documented up-front so the user can review):
 - DataLoader: we reuse the battle-tested PyTorch `TextAudioSpeakerLoader` +
   `TextAudioSpeakerCollate` + `DistributedBucketSampler` from upstream and
   convert tensors to mx.array at the batch boundary. Writing a 100%-native
   MLX loader would duplicate upstream's bucket sampler logic; the I/O
   bottleneck is .pt file loading (PyTorch workers handle it well), not
   tensor format. The Phase 1+2 prototype hinted at this approach.
 - LR scheduler: manual ExponentialLR (gamma=lr_decay), applied per-step.
 - Optimizer: two AdamW instances, one for net_g, one for net_d. text-low-lr
   implemented as per-parameter LR scaling on text_embedding/encoder_text/mrte.
 - Ckpt format: native MLX safetensors per-epoch (init_g/init_d pattern).
   convert_full.py export-g can round-trip to PyTorch .pth for fallback
   inference via the existing GPT-SoVITS pipeline.

Usage:
  python train_r6.py --dry-run   # 5 steps, verify everything connects
  python train_r6.py             # full 5-epoch run

Outputs (in --output-dir, default ~/tts/_sovits_mlx_train/r6/):
  e{N}_g.safetensors        net_g state at end of epoch N (17..21)
  e{N}_d.safetensors        net_d state
  train.log                 line-per-step loss log
  meta.json                 run metadata (config, timing, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Ensure both the MLX port and GPT-SoVITS are importable
sys.path.insert(0, "/Users/kaede/tts/_sovits_mlx")
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

# MLX port modules
from models import SynthesizerTrn, default_config
from discriminator import MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss as kl_loss_fn
import commons as _commons
import mel as mlx_mel

# Upstream PyTorch — used only for the DataLoader
import torch
from torch.utils.data import DataLoader
from module.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)


# ---------------------------------------------------------------------------
# Config — matches r5 except c_kl=1.0 (upstream default, restored from r5 patch)
# ---------------------------------------------------------------------------
R6_CONFIG = SimpleNamespace(
    version="v2ProTw",
    # train
    seed=1234,
    epochs=5,                              # e17..e21
    starting_epoch=17,                     # next after r5 e16
    save_every_epoch=1,
    learning_rate=1e-5,
    betas=(0.8, 0.99),
    eps=1e-9,
    batch_size=8,                          # user-requested sweet spot
    batch_size_fallback=6,                 # if OOM at 8
    lr_decay=0.999875,
    segment_size=20480,                    # audio samples
    c_mel=45.0,
    c_kl=1.0,                              # THE WHOLE POINT — r5 had to use 0.0
    text_low_lr_rate=0.4,                  # text_embedding/encoder_text/mrte
    log_interval=10,
    # data
    exp_dir="/Users/kaede/tts/tw_finetune_synthetic",
    max_wav_value=32768.0,
    sampling_rate=32000,
    filter_length=2048,
    hop_length=640,
    win_length=2048,
    n_mel_channels=128,
    mel_fmin=0.0,
    mel_fmax=None,
)

SEG_FRAMES = R6_CONFIG.segment_size // R6_CONFIG.hop_length   # 32


# ---------------------------------------------------------------------------
# DataLoader wrapper: PyTorch DataLoader → MLX iter
# ---------------------------------------------------------------------------
def build_loader(cfg, version, batch_size):
    """Build the upstream PyTorch DataLoader + bucket sampler."""
    data_hps = SimpleNamespace(
        exp_dir=cfg.exp_dir,
        max_wav_value=cfg.max_wav_value,
        sampling_rate=cfg.sampling_rate,
        filter_length=cfg.filter_length,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )
    dataset = TextAudioSpeakerLoader(data_hps, version=version)
    sampler = DistributedBucketSampler(
        dataset, batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
         1400, 1500, 1600, 1700, 1800, 1900],
        num_replicas=1, rank=0, shuffle=True,
    )
    collate = TextAudioSpeakerCollate(version=version)
    loader = DataLoader(
        dataset, num_workers=4, shuffle=False, pin_memory=False,
        collate_fn=collate, batch_sampler=sampler, persistent_workers=True,
        prefetch_factor=2,
    )
    return dataset, loader, sampler


def to_mx(batch, version):
    """Convert one upstream-collated batch to mx.array."""
    if version in {"v2Pro", "v2ProPlus", "v2ProTw", "v2ProPlusTw"}:
        ssl_p, ssl_l, spec_p, spec_l, wav_p, wav_l, text_p, text_l, sv_emb = batch
    else:
        raise NotImplementedError("v1/v2 not used for r6")
    # All inputs to MLX are float32 (or int32 for ids/lengths). Squeeze ssl B,1,C,T → B,C,T.
    return dict(
        ssl=mx.array(ssl_p.numpy().astype(np.float32)),
        ssl_lengths=mx.array(ssl_l.numpy().astype(np.int32)),
        spec=mx.array(spec_p.numpy().astype(np.float32)),
        spec_lengths=mx.array(spec_l.numpy().astype(np.int32)),
        wav=mx.array(wav_p.numpy().astype(np.float32)),
        wav_lengths=mx.array(wav_l.numpy().astype(np.int32)),
        text=mx.array(text_p.numpy().astype(np.int32)),
        text_lengths=mx.array(text_l.numpy().astype(np.int32)),
        sv_emb=mx.array(sv_emb.numpy().astype(np.float32)),
    )


# ---------------------------------------------------------------------------
# LR helpers
# ---------------------------------------------------------------------------
def lr_at_step(base_lr: float, gamma: float, global_step: int) -> float:
    """Exponential LR decay applied per step.  Mirrors PyTorch ExponentialLR."""
    return base_lr * (gamma ** global_step)


def scale_text_low_lr_grads(grads_tree, text_low_lr_rate: float):
    """Mutate the grads tree so that text_embedding/encoder_text/mrte receive
    grads scaled by text_low_lr_rate. Implemented by recursively walking the
    flat key/grad pairs and rebuilding the nested tree.

    Faster: scale in-place on the flat tuple list before optimizer.update.
    """
    flat = tree_flatten(grads_tree)
    scaled = []
    for k, g in flat:
        if any(s in k for s in (
            "enc_p.text_embedding.",
            "enc_p.encoder_text.",
            "enc_p.mrte.",
        )):
            g = g * text_low_lr_rate
        scaled.append((k, g))
    from mlx.utils import tree_unflatten
    return tree_unflatten(scaled)


# ---------------------------------------------------------------------------
# Loss functions (factored for value_and_grad)
# ---------------------------------------------------------------------------
def g_full_loss(net_g, net_d, batch, cfg, slice_key):
    """Single-batch generator loss. Returns (loss_total, components_dict)."""
    y_hat, kl_ssl, ids_slice, x_mask, z_mask, stats, quantized = net_g(
        batch["ssl"], batch["spec"], batch["spec_lengths"],
        batch["text"], batch["text_lengths"], batch["sv_emb"],
        segment_size_frames=SEG_FRAMES, slice_key=slice_key,
    )
    z, z_p, m_p, logs_p, m_q, logs_q = stats

    # KL closed-form
    loss_kl = kl_loss_fn(z_p, logs_q, m_p, logs_p, z_mask) * cfg.c_kl

    # Slice real audio at the SAME ids
    y_real_slice = _commons.slice_segments(
        batch["wav"], ids_slice * cfg.hop_length, SEG_FRAMES * cfg.hop_length,
    )

    # Mel L1 on the slice (use spec_to_mel for real if possible; simpler: stft both)
    mel_real = mlx_mel.mel_spectrogram(
        y_real_slice[:, 0, :], cfg.filter_length, cfg.n_mel_channels,
        cfg.sampling_rate, cfg.hop_length, cfg.win_length, cfg.mel_fmin, cfg.mel_fmax,
        center=False,
    )
    mel_fake = mlx_mel.mel_spectrogram(
        y_hat[:, 0, :], cfg.filter_length, cfg.n_mel_channels,
        cfg.sampling_rate, cfg.hop_length, cfg.win_length, cfg.mel_fmin, cfg.mel_fmax,
        center=False,
    )
    F_common = min(mel_real.shape[-1], mel_fake.shape[-1])
    loss_mel = mx.mean(mx.abs(mel_real[..., :F_common] - mel_fake[..., :F_common])) * cfg.c_mel

    # Discriminator → adversarial + feature matching
    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_real_slice, y_hat)
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, _ = generator_loss(y_d_hat_g)

    loss_total = loss_gen + loss_fm + loss_mel + kl_ssl * 1.0 + loss_kl
    return loss_total, (loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl)


def d_only_loss(net_d, y_real_slice, y_hat_detached):
    y_d_hat_r, y_d_hat_g, _, _ = net_d(y_real_slice, y_hat_detached)
    loss_d, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
    return loss_d


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------
def train(args):
    cfg = R6_CONFIG
    if args.batch_size:
        cfg.batch_size = args.batch_size
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "train.log"
    meta_path = Path(args.output_dir) / "meta.json"

    # ---- reproducibility ----
    mx.random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- build models ----
    print("Building net_g and net_d ...", flush=True)
    net_g = SynthesizerTrn(**default_config())
    net_d = MultiPeriodDiscriminator(version=cfg.version)
    mx.eval(net_g.parameters(), net_d.parameters())

    # ---- load r5 e16 ----
    print(f"Loading {args.init_g} ...", flush=True)
    net_g.load_weights(args.init_g, strict=True)
    print(f"Loading {args.init_d} ...", flush=True)
    net_d.load_weights(args.init_d, strict=True)
    mx.eval(net_g.parameters(), net_d.parameters())
    print("  resumed from r5 e16.")

    # ---- optimizer ----
    opt_g = optim.AdamW(learning_rate=cfg.learning_rate, betas=list(cfg.betas),
                        eps=cfg.eps, weight_decay=0.0)
    opt_d = optim.AdamW(learning_rate=cfg.learning_rate, betas=list(cfg.betas),
                        eps=cfg.eps, weight_decay=0.0)

    # ---- value_and_grad fns ----
    def loss_g_only(model_g, batch, slice_key):
        return g_full_loss(model_g, net_d, batch, cfg, slice_key)[0]

    g_grad_fn = nn.value_and_grad(net_g, loss_g_only)

    def loss_d_only(model_d, y_real, y_fake_detached):
        return d_only_loss(model_d, y_real, y_fake_detached)

    d_grad_fn = nn.value_and_grad(net_d, loss_d_only)

    # ---- data ----
    print(f"Building DataLoader (bs={cfg.batch_size}) ...", flush=True)
    dataset, loader, sampler = build_loader(cfg, cfg.version, cfg.batch_size)
    print(f"  dataset rows: {len(dataset)}")

    # ---- meta ----
    meta = dict(
        cfg=cfg.__dict__,
        dataset_rows=len(dataset),
        init_g=args.init_g, init_d=args.init_d,
        seg_frames=SEG_FRAMES,
        starting_epoch=cfg.starting_epoch,
        dry_run=args.dry_run,
        bench_mlx_version=mx.__version__ if hasattr(mx, "__version__") else "?",
    )
    meta_path.write_text(json.dumps(meta, indent=2))

    global_step = 0
    log_fh = open(log_path, "a", buffering=1)
    log_fh.write(f"# r6 launched {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fh.write(f"# cfg={cfg.__dict__}\n")
    t_start = time.time()

    # ---- epochs ----
    for epoch_offset in range(cfg.epochs):
        epoch = cfg.starting_epoch + epoch_offset
        sampler.set_epoch(epoch)

        t_epoch_start = time.time()
        comp_acc = {"gen": 0., "fm": 0., "mel": 0., "kl_ssl": 0., "kl": 0., "d": 0.}
        n_batches = 0
        for batch_idx, batch_pt in enumerate(loader):
            batch = to_mx(batch_pt, cfg.version)

            # current LR
            lr = lr_at_step(cfg.learning_rate, cfg.lr_decay, global_step)
            opt_g.learning_rate = lr
            opt_d.learning_rate = lr

            # ---- D step (using y_hat detached) ----
            # Compute y_hat once (no grad)
            slice_key = mx.random.key(global_step + 7777)
            yh, kl_ssl, ids_slice, _, z_mask, stats, _ = net_g(
                batch["ssl"], batch["spec"], batch["spec_lengths"],
                batch["text"], batch["text_lengths"], batch["sv_emb"],
                segment_size_frames=SEG_FRAMES, slice_key=slice_key,
            )
            y_real_slice = _commons.slice_segments(
                batch["wav"], ids_slice * cfg.hop_length, SEG_FRAMES * cfg.hop_length,
            )
            y_hat_detached = mx.stop_gradient(yh)
            loss_d_val, grads_d = d_grad_fn(net_d, y_real_slice, y_hat_detached)
            opt_d.update(net_d, grads_d)
            mx.eval(net_d.parameters(), opt_d.state, loss_d_val)

            # ---- G step (full loss with KL) ----
            loss_g_val, grads_g = g_grad_fn(net_g, batch, slice_key)
            # text_low_lr scaling
            grads_g = scale_text_low_lr_grads(grads_g, cfg.text_low_lr_rate)
            opt_g.update(net_g, grads_g)
            mx.eval(net_g.parameters(), opt_g.state, loss_g_val)

            # Component breakdown for logging (cheap forward, no grad)
            _, comps = g_full_loss(net_g, net_d, batch, cfg, slice_key)
            l_gen, l_fm, l_mel, l_kls, l_kl = (float(x) for x in comps)
            l_d = float(loss_d_val)
            comp_acc["gen"] += l_gen; comp_acc["fm"] += l_fm; comp_acc["mel"] += l_mel
            comp_acc["kl_ssl"] += l_kls; comp_acc["kl"] += l_kl; comp_acc["d"] += l_d
            n_batches += 1

            if global_step % cfg.log_interval == 0:
                line = (f"epoch={epoch} step={global_step} lr={lr:.3e} "
                        f"gen={l_gen:.3f} fm={l_fm:.3f} mel={l_mel:.3f} "
                        f"kl_ssl={l_kls:.4f} kl={l_kl:.4f} d={l_d:.3f}")
                print(line, flush=True)
                log_fh.write(line + "\n")
            global_step += 1

            if args.dry_run and batch_idx + 1 >= args.dry_run_steps:
                break

        t_epoch = time.time() - t_epoch_start
        avg = {k: v / max(n_batches, 1) for k, v in comp_acc.items()}
        summary = (
            f"== epoch {epoch} done == "
            f"batches={n_batches} time={t_epoch:.1f}s "
            f"avg(gen)={avg['gen']:.3f} avg(fm)={avg['fm']:.3f} avg(mel)={avg['mel']:.3f} "
            f"avg(kl_ssl)={avg['kl_ssl']:.4f} avg(kl)={avg['kl']:.4f} avg(d)={avg['d']:.3f}"
        )
        print(summary, flush=True)
        log_fh.write(summary + "\n")

        # ---- save ----
        if (epoch_offset + 1) % cfg.save_every_epoch == 0:
            ckpt_g = Path(args.output_dir) / f"e{epoch}_g.safetensors"
            ckpt_d = Path(args.output_dir) / f"e{epoch}_d.safetensors"
            # net_g.save_weights uses MLX's safetensors writer
            net_g.save_weights(str(ckpt_g))
            net_d.save_weights(str(ckpt_d))
            print(f"  saved {ckpt_g.name}, {ckpt_d.name}", flush=True)
            log_fh.write(f"# saved e{epoch}\n")

        if args.dry_run:
            print("Dry-run complete.", flush=True)
            break

    log_fh.write(f"# total wall-clock: {time.time() - t_start:.1f}s\n")
    log_fh.close()
    print("r6 training complete.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-g",
                    default="/Users/kaede/tts/_sovits_mlx_train/r6/init_g.safetensors")
    ap.add_argument("--init-d",
                    default="/Users/kaede/tts/_sovits_mlx_train/r6/init_d.safetensors")
    ap.add_argument("--output-dir",
                    default="/Users/kaede/tts/_sovits_mlx_train/r6")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="override R6_CONFIG.batch_size (0 keeps default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run --dry-run-steps batches in first epoch, then exit")
    ap.add_argument("--dry-run-steps", type=int, default=5)
    args = ap.parse_args()
    train(args)
