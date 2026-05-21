"""Per-module shape + numerical sanity tests.

Loads the original PyTorch S2 model and the converted MLX model, feeds the
same input through equivalent submodules, and reports max-abs diff.
"""

import os, sys
import numpy as np
import mlx.core as mx

# Add repo paths so PyTorch model can import
ROOT = "/Users/kaede/tts"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/GPT-SoVITS")
sys.path.insert(0, f"{ROOT}/GPT-SoVITS/GPT_SoVITS")
sys.path.insert(0, f"{ROOT}/_sovits_mlx")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import torch

from GPT_SoVITS.module.models import SynthesizerTrn as PT_SynthesizerTrn
from safetensors.numpy import load_file
from models import SynthesizerTrn as MLX_SynthesizerTrn, default_config


CKPT_PATH = f"{ROOT}/tw_finetune_synthetic/s2_logs_r4/s2_full_15.pth"


def torch_to_np(t):
    return t.detach().cpu().numpy()


def report(name, pt_np, mlx_arr):
    if isinstance(mlx_arr, mx.array):
        mlx_np = np.array(mlx_arr)
    else:
        mlx_np = mlx_arr
    diff = np.abs(pt_np - mlx_np)
    norm = max(float(np.abs(pt_np).mean()), 1e-9)
    rel = diff.mean() / norm
    print(f"[{name}]  PT shape {pt_np.shape}   MLX shape {mlx_np.shape}   "
          f"max_abs={diff.max():.3e}   mean_abs={diff.mean():.3e}   rel={rel:.3e}")


def load_pt():
    print("Loading PT SynthesizerTrn ...", flush=True)
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    # Build a hps-equivalent
    pt_kwargs = dict(
        spec_channels=1025,
        segment_size=32,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        n_speakers=300,
        gin_channels=1024,
        semantic_frame_rate="25hz",
        freeze_quantizer=True,
        version="v2ProTw",
    )
    pt = PT_SynthesizerTrn(**pt_kwargs)
    sd = ck["model"]
    missing, unexpected = pt.load_state_dict(sd, strict=False)
    print(f"  PT load: missing={len(missing)} unexpected={len(unexpected)}")
    pt.eval()
    return pt


def load_mlx():
    print("Loading MLX SynthesizerTrn ...", flush=True)
    m = MLX_SynthesizerTrn(**default_config())
    sf = load_file(f"{ROOT}/_sovits_mlx/model.safetensors")
    m.load_weights([(k, mx.array(v)) for k, v in sf.items()], strict=True)
    mx.eval(m.parameters())
    return m


def test_ref_enc(pt, mlx_m):
    """ref_enc takes (B, 704, T_ref) spec -> (B, 1024, 1) speaker embedding."""
    np.random.seed(0)
    x = np.random.randn(1, 1025, 60).astype(np.float32)
    # PT path
    with torch.no_grad():
        ge_pt = pt.ref_enc(torch.from_numpy(x[:, :704]) * 1.0, None)
    # MLX path
    ge_mlx = mlx_m.ref_enc(mx.array(x[:, :704]))
    report("ref_enc", torch_to_np(ge_pt), ge_mlx)


def test_sv_emb_ge(pt, mlx_m):
    """Test the v2Pro embedding path: ge_raw + sv_emb -> prelu(.)"""
    np.random.seed(1)
    spec = np.random.randn(1, 1025, 60).astype(np.float32)
    sv = np.random.randn(1, 20480).astype(np.float32)
    with torch.no_grad():
        ge_raw = pt.ref_enc(torch.from_numpy(spec[:, :704]), None)
        sv_pt = pt.sv_emb(torch.from_numpy(sv))
        ge_pt = pt.prelu(ge_raw + sv_pt.unsqueeze(-1))
    ge_mlx = mlx_m._compute_ge(mx.array(spec), mx.array(sv))
    report("ge (incl sv_emb, prelu)", torch_to_np(ge_pt), ge_mlx)


def test_ge_to_512(pt, mlx_m):
    """ge_to512 -> ge_512."""
    np.random.seed(2)
    ge = np.random.randn(1, 1024, 1).astype(np.float32)
    with torch.no_grad():
        ge_512_pt = pt.ge_to512(torch.from_numpy(ge).transpose(2, 1)).transpose(2, 1)
    ge_mlx = mlx_m.ge_to512(mx.array(ge).transpose(0, 2, 1)).transpose(0, 2, 1)
    report("ge_to512", torch_to_np(ge_512_pt), ge_mlx)


def test_quantizer_decode(pt, mlx_m):
    np.random.seed(3)
    codes = np.random.randint(0, 1024, size=(1, 1, 30)).astype(np.int64)
    with torch.no_grad():
        q_pt = pt.quantizer.decode(torch.from_numpy(codes))
    q_mlx = mlx_m.quantizer.decode(mx.array(codes))
    report("quantizer.decode", torch_to_np(q_pt), q_mlx)


def test_dec(pt, mlx_m):
    """Decoder generator: (B, 192, T_y) + ge -> (B, 1, T_wav)."""
    np.random.seed(4)
    z = np.random.randn(1, 192, 30).astype(np.float32)
    ge = np.random.randn(1, 1024, 1).astype(np.float32)
    with torch.no_grad():
        wav_pt = pt.dec(torch.from_numpy(z), g=torch.from_numpy(ge))
    wav_mlx = mlx_m.dec(mx.array(z), g=mx.array(ge))
    report("dec(Generator)", torch_to_np(wav_pt), wav_mlx)


def test_flow(pt, mlx_m):
    """Flow reverse: (B, 192, T) + ge -> (B, 192, T)."""
    np.random.seed(5)
    z_p = np.random.randn(1, 192, 30).astype(np.float32)
    mask = np.ones((1, 1, 30), dtype=np.float32)
    ge = np.random.randn(1, 1024, 1).astype(np.float32)
    with torch.no_grad():
        z_pt = pt.flow(torch.from_numpy(z_p), torch.from_numpy(mask),
                       g=torch.from_numpy(ge), reverse=True)
    z_mlx = mlx_m.flow(mx.array(z_p), mx.array(mask), g=mx.array(ge), reverse=True)
    report("flow(reverse)", torch_to_np(z_pt), z_mlx)


def test_enc_p_full(pt, mlx_m):
    """TextEncoder (enc_p) — quant + text + ge_512 -> y, m_p, logs_p."""
    np.random.seed(6)
    T_y = 30
    T_t = 25
    quantized = np.random.randn(1, 768, T_y).astype(np.float32)
    text = np.random.randint(0, 1033, size=(1, T_t)).astype(np.int64)
    ge_512 = np.random.randn(1, 512, 1).astype(np.float32)
    y_lengths_pt = torch.LongTensor([T_y])
    text_lengths_pt = torch.LongTensor([T_t])
    with torch.no_grad():
        x_pt, m_pt, logs_pt, ymask_pt, _, _ = pt.enc_p(
            torch.from_numpy(quantized), y_lengths_pt,
            torch.from_numpy(text), text_lengths_pt,
            torch.from_numpy(ge_512),
        )
    _, m_mlx, logs_mlx, _ = mlx_m.enc_p(
        mx.array(quantized),
        mx.array(np.array([T_y], dtype=np.int32)),
        mx.array(text),
        mx.array(np.array([T_t], dtype=np.int32)),
        mx.array(ge_512),
    )
    report("enc_p.m_p", torch_to_np(m_pt), m_mlx)
    report("enc_p.logs_p", torch_to_np(logs_pt), logs_mlx)


if __name__ == "__main__":
    pt = load_pt()
    mlx_m = load_mlx()
    print()
    test_ref_enc(pt, mlx_m)
    test_sv_emb_ge(pt, mlx_m)
    test_ge_to_512(pt, mlx_m)
    test_quantizer_decode(pt, mlx_m)
    test_dec(pt, mlx_m)
    test_flow(pt, mlx_m)
    test_enc_p_full(pt, mlx_m)
    print("\nDone.")
