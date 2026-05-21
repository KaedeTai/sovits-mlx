"""Backward-pass smoke test.

Verifies that MLX autograd through:
  Generator → audio
  Audio → mel via mlx_mel
  L1 mel loss
produces finite gradients with shapes matching the trainable parameters.

Also exercises the discriminator backward path.
"""

import sys
sys.path.insert(0, "/Users/kaede/tts/_sovits_mlx")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from hifigan import Generator
from discriminator import MultiPeriodDiscriminator
from posterior_encoder import PosteriorEncoder
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
import mel as mlx_mel


def shape_eq(t1, t2):
    return tuple(t1.shape) == tuple(t2.shape)


def assert_finite_grad_shapes(model, grads):
    """Walk a parallel tree of params/grads, ensure shapes match & finite."""
    flat_p = dict(tree_flatten(model.parameters()))
    flat_g = dict(tree_flatten(grads))
    missing, mismatched, nan_keys = [], [], []
    for k, p in flat_p.items():
        g = flat_g.get(k)
        if g is None:
            missing.append(k); continue
        if tuple(g.shape) != tuple(p.shape):
            mismatched.append((k, tuple(p.shape), tuple(g.shape)))
        if not bool(mx.all(mx.isfinite(g))):
            nan_keys.append(k)
    return missing, mismatched, nan_keys


def test_generator_grad():
    print("\n=== Generator backward through mel L1 ===")
    g = Generator(
        initial_channel=192, resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        gin_channels=1024,
    )
    mx.eval(g.parameters())

    B = 1
    T_lat = 32                  # generator input length (segment_size=20480 / hop=640)
    z = mx.random.normal((B, 192, T_lat))
    cond = mx.random.normal((B, 1024, 1))

    # Measure actual upsample (=640 for v2ProTw).
    audio_dummy = g(z, g=cond)
    T_audio = audio_dummy.shape[-1]
    audio_target = mx.random.normal((B, T_audio))

    def loss_fn(model):
        audio = model(z, g=cond)             # (B, 1, T_wav)
        audio = audio[:, 0, :]                # (B, T_wav)
        mel_g = mlx_mel.mel_spectrogram(
            audio, n_fft=2048, num_mels=128, sampling_rate=32000,
            hop_size=640, win_size=2048, fmin=0.0, fmax=None, center=False,
        )
        mel_t = mlx_mel.mel_spectrogram(
            audio_target, n_fft=2048, num_mels=128, sampling_rate=32000,
            hop_size=640, win_size=2048, fmin=0.0, fmax=None, center=False,
        )
        return mx.mean(mx.abs(mel_g - mx.stop_gradient(mel_t)))

    loss, grads = nn.value_and_grad(g, loss_fn)(g)
    mx.eval(loss, grads)
    print(f"  loss = {float(loss):.6f}")
    missing, mismatched, nan_keys = assert_finite_grad_shapes(g, grads)
    print(f"  missing grads: {len(missing)}")
    print(f"  shape mismatches: {len(mismatched)}")
    print(f"  non-finite grads: {len(nan_keys)}")
    if mismatched:
        print("    mismatched (first 3):", mismatched[:3])
    if nan_keys:
        print("    nan keys (first 3):", nan_keys[:3])
    assert len(mismatched) == 0
    assert len(nan_keys) == 0
    n_total = len(list(tree_flatten(g.parameters())))
    print(f"  params traversed: {n_total}, all OK")
    return True


def test_discriminator_grad():
    print("\n=== Discriminator backward (LSGAN) ===")
    d = MultiPeriodDiscriminator(version="v2ProTw")
    mx.eval(d.parameters())

    B, T = 1, 20480
    y = mx.random.normal((B, 1, T))
    y_hat = mx.random.normal((B, 1, T))

    def loss_fn(model):
        rs, gs, _, _ = model(y, mx.stop_gradient(y_hat))
        l, _, _ = discriminator_loss(rs, gs)
        return l

    loss, grads = nn.value_and_grad(d, loss_fn)(d)
    mx.eval(loss, grads)
    print(f"  loss = {float(loss):.6f}")
    missing, mismatched, nan_keys = assert_finite_grad_shapes(d, grads)
    print(f"  missing: {len(missing)}  mismatch: {len(mismatched)}  nan: {len(nan_keys)}")
    assert len(mismatched) == 0
    assert len(nan_keys) == 0
    n_total = len(list(tree_flatten(d.parameters())))
    print(f"  params traversed: {n_total}, all OK")
    return True


def test_posterior_encoder_grad():
    print("\n=== PosteriorEncoder backward ===")
    enc = PosteriorEncoder(in_channels=1025, out_channels=192, hidden_channels=192,
                            kernel_size=5, dilation_rate=1, n_layers=16,
                            gin_channels=1024)
    mx.eval(enc.parameters())
    B, T = 1, 32
    x = mx.random.normal((B, 1025, T))
    lens = mx.array([T], dtype=mx.int32)
    g = mx.random.normal((B, 1024, 1))

    def loss_fn(model):
        z, mn, logs, mask = model(x, lens, g=g, noise=mx.zeros((B, 192, T)))
        return mx.mean(z * z) + mx.mean(logs * logs)

    loss, grads = nn.value_and_grad(enc, loss_fn)(enc)
    mx.eval(loss, grads)
    print(f"  loss = {float(loss):.6f}")
    missing, mismatched, nan_keys = assert_finite_grad_shapes(enc, grads)
    print(f"  missing: {len(missing)}  mismatch: {len(mismatched)}  nan: {len(nan_keys)}")
    assert len(mismatched) == 0
    assert len(nan_keys) == 0
    return True


if __name__ == "__main__":
    test_posterior_encoder_grad()
    test_discriminator_grad()
    test_generator_grad()
    print("\nAll backward tests passed.")
