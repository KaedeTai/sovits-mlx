# MLX training port — Phase 1 + 2 status

## What landed this session

```
posterior_encoder.py     enc_q (spec → z, m_q, logs_q); WN reuse
discriminator.py         MPD + MSD + DiscriminatorP/S; 7 periods for v2Pro
losses.py                generator_loss, discriminator_loss, feature_loss, kl_loss
mel.py                   STFT (manual frame+rfft) + librosa mel filterbank
train_step.py            single-step G+D prototype + 10-step speed benchmark
tests/test_grad.py       autograd smoke test (PostEnc, MPD, Generator+mel)
tests/test_parity.py     PyTorch → MLX weight transfer, 1-step loss comparison
```

## Correctness

| check | result |
|------|--------|
| losses parity vs PyTorch CPU | disc/gen/feature/kl all within 1e-6 relative |
| mel spectrogram parity | spec rel 5e-7, mel rel 5e-4 (acceptable; log of small values) |
| discriminator parity (same weights, random input) | rel ~3e-6 across all 8 sub-discs |
| autograd through Generator+mel L1+LSGAN | finite, shapes match, 0 NaN |
| autograd through Discriminator (LSGAN) | finite, shapes match, 0 NaN |
| autograd through PosteriorEncoder | finite, shapes match, 0 NaN |
| 1-step G_total parity (PyTorch CPU vs MLX, same weights) | 169.84 vs 170.52, rel 4.0e-3 |

The 4e-3 G_total gap is dominated by mel L1 (rel 4.2e-3), which comes from
the linear→mel matmul + log accumulating float32 noise. gen/disc head numbers
are essentially identical (1e-7).

## Speed (the answer you actually want)

10-step training step (G update + D update), batch=4, segment=20480, 2-step warmup,
per-step fresh input — multiple back-to-back trials:

| backend  | sec/step (trial 1) | trial 2 | trial 3 |
|----------|-------------------|---------|---------|
| MLX      | 0.356             | 0.516   | 0.524   |
| PT-MPS   | 0.454             | 0.471   | 0.479   |

MLX has higher variance (likely r5 GPU contention on some runs); PT-MPS is steady.
With kernels warm on both sides, **PyTorch-MPS and MLX are within 10% of each other**.
Median MLX ≈ 0.52 s/step; median PT-MPS ≈ 0.47 s/step. PyTorch is slightly ahead.

**The earlier ad-hoc measurement that showed MLX 2× faster was a PyTorch
first-step compilation overhead artifact — not a real win.**

## Honest assessment: is an MLX training port a clear win?

**No, not a clear win on speed.** On the dominant compute (Generator + Discriminator
forward + backward + AdamW + mel L1), MLX and PyTorch-MPS are roughly tied,
with PT-MPS slightly faster in steady state.

What MLX *does* offer:
1. Same backend as the inference stack we already ship → operational uniformity.
2. Tighter memory management on Apple Silicon (we haven't measured this).
3. No "MPS-only op missing" papercuts (KL backward, autograd through some
   non-contiguous splits — see the r5 c_kl=0.0 workaround in pathM_r5.json).
4. Latitude to experiment with bf16 training (MLX bf16 is more mature than MPS bf16).

Why r5's reported 0.36 batch/sec is so far below the 2-3 batch/sec gen+disc number:
the slimmed step here excludes data loading, TextEncoder, PosteriorEncoder, Flow,
and the second discriminator pass during training. Those make up most of r5's
wall time. **Speeding up gen+disc by 10% would only move the needle a little;
to make a real dent you'd need to port the entire pipeline AND find an MLX edge
on those other ops too.**

## Blockers found that would affect Phase 3

1. `mlx.pad` reflect mode: works in MLX 0.21+; older builds need numpy fallback.
   `mel.py` handles this with a try/except. Should we pin mlx>=0.21 in setup?
2. PyTorch `weight_norm` parameterization: not present in MLX. Currently we
   store materialized weights; load/save needs a converter that collapses
   `weight_g`/`weight_v` on input and re-expands on output (the latter is
   only needed if we want PyTorch to consume MLX-trained checkpoints).
3. `torch.stft` has no MLX equivalent. Manual frame+rfft works fine on the
   training-spec side (verified rel 5e-7 vs PyTorch). For training, the spec
   is precomputed once per epoch by the dataloader so this is not a hot path.
4. AdamW: MLX `optimizers.AdamW` matches PyTorch's signature. No issues.
5. **No KL-loss MPS issue** because MLX has its own backward and the existing
   r5 c_kl=0 workaround is not needed. Mild positive.

## Recommendation for Phase 3

Given the speed parity, the strongest argument for Phase 3 is operational
consistency with the inference stack and avoiding MPS papercuts. But it's
**not** the path to a faster r5.

If speed is the goal, the higher-yield investigations are:
- Profile where r5 actually spends its 2.78 sec/step (data loading vs model
  forward vs backward), since the gen+disc compute is only ~0.47s/step.
- Look at batch_size=8 — at batch=4 we may be memory-bound rather than
  compute-bound on MPS.
- Try MLX bf16 on the whole pipeline — that's where the real headroom is.

If Phase 3 ships anyway, the missing pieces are:
- TextEncoder MLX backward (already exists for forward; just needs autograd
  to be exercised — should "just work" since it's the same ops as the
  Encoder used in inference).
- Flow MLX backward (same).
- Dataloader port (numpy/torch DataLoader → mx.array per batch is fine).
- Checkpoint compat: PyTorch ckpt ↔ MLX ckpt via the existing convert.py
  pattern, extended to discriminator + posterior + optimizer state.
- LR scheduler (lr_decay=0.999875, exponential).

These are bookwork, not architectural — should be 2-3 days once we know we
want to do them. The decision blocker is whether the speed argument holds.
