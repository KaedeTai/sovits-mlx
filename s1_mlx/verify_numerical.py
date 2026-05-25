"""Numerical-equivalence test for the S1 MLX port.

Run from this directory:
    python verify_numerical.py
"""
from __future__ import annotations
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS")
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")
sys.path.insert(0, "/Users/kaede/tts/_sovits_mlx/s1_mlx")

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from model import T2SModel
from safetensors.numpy import load_file

SAFETENSORS = "/Users/kaede/tts/_sovits_mlx/s1_mlx/s1.safetensors"
CONFIG      = "/Users/kaede/tts/_sovits_mlx/s1_mlx/s1.config.json"
CKPT        = "/Users/kaede/tts/_s1_trilingual/arm_A_e15_trilingual.ckpt"


def build_pt():
    cfg = json.load(open(CONFIG))
    pt = Text2SemanticLightningModule(cfg, "****", is_train=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    pt.load_state_dict(ck["weight"])
    return pt.eval().model


def build_mlx():
    cfg = json.load(open(CONFIG))
    m = T2SModel(cfg)
    sd = load_file(SAFETENSORS)
    m.load_weights([(k, mx.array(v.astype(np.float32))) for k, v in sd.items()], strict=True)
    mx.eval(m.parameters())
    return m


def pt_prefill_logits(ptm, phones, bert, prompt):
    with torch.no_grad():
        x = ptm.ar_text_embedding(phones)
        x = x + ptm.bert_proj(bert.transpose(1, 2))
        x = ptm.ar_text_position(x)
        y_emb = ptm.ar_audio_embedding(prompt)
        y_pos = ptm.ar_audio_position(y_emb)
        xy_pos = torch.cat([x, y_pos], dim=1)
        x_len = x.shape[1]; y_len = y_emb.shape[1]
        src_len = x_len + y_len
        bsz = 1
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
        y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                            (x_len, 0), value=False)
        xy_attn_mask = (torch.cat([x_attn_mask_pad, y_attn_mask], dim=0).unsqueeze(0)
                        .expand(bsz * ptm.num_head, -1, -1)
                        .view(bsz, ptm.num_head, src_len, src_len).bool())
        xy_dec, kc, vc = ptm.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
        logits = ptm.ar_predict_layer(xy_dec[:, -1])
        return logits, xy_dec, kc, vc, y_len


def mlx_prefill_logits(m, phones, bert, prompt):
    phones_mx = mx.array(phones.numpy().astype(np.int32))
    bert_mx = mx.array(bert.numpy())
    prompt_mx = mx.array(prompt.numpy().astype(np.int32))
    text_emb = m.make_text_prefix(phones_mx, bert_mx)
    hidden, kcs, vcs, _ = m.prefill(text_emb, prompt_mx)
    logits = m.logits(hidden)
    return logits, hidden, kcs, vcs, prompt_mx.shape[1]


def test_prefill():
    print("\n=== TEST 1: Prefill numerical equivalence ===")
    ptm = build_pt(); m = build_mlx()
    np.random.seed(0); torch.manual_seed(0)
    T_t, T_p = 30, 50
    phones = torch.randint(0, 1033, (1, T_t), dtype=torch.long)
    bert = torch.randn(1, 1024, T_t, dtype=torch.float32)
    prompt = torch.randint(0, 1024, (1, T_p), dtype=torch.long)

    lp, _, _, _, _ = pt_prefill_logits(ptm, phones, bert, prompt)
    lm, _, _, _, _ = mlx_prefill_logits(m, phones, bert, prompt)
    a = lp[0].numpy().astype(np.float32); b = np.array(lm[0]).astype(np.float32)
    abs_d = np.abs(a - b); rel_d = abs_d / (np.abs(a) + 1e-6)
    print(f"  logits shape: PT={tuple(lp.shape)} MLX={tuple(lm.shape)}")
    print(f"  max abs diff:  {abs_d.max():.6f}")
    print(f"  mean abs diff: {abs_d.mean():.6f}")
    print(f"  max rel diff:  {rel_d.max():.6f}")
    print(f"  argmax match: {a.argmax() == b.argmax()}")
    top10_pt = set(np.argsort(a)[::-1][:10].tolist())
    top10_mx = set(np.argsort(b)[::-1][:10].tolist())
    print(f"  top-10 set match: {top10_pt == top10_mx}")
    assert a.argmax() == b.argmax(), "argmax mismatch on prefill"
    return ptm, m


def test_rollout(ptm, m, n_steps=8):
    print(f"\n=== TEST 2: {n_steps}-step greedy rollout ===")
    np.random.seed(0); torch.manual_seed(0)
    T_t, T_p = 30, 50
    phones = torch.randint(0, 1033, (1, T_t), dtype=torch.long)
    bert = torch.randn(1, 1024, T_t, dtype=torch.float32)
    prompt = torch.randint(0, 1024, (1, T_p), dtype=torch.long)

    # PT rollout
    with torch.no_grad():
        x = ptm.ar_text_embedding(phones); x = x + ptm.bert_proj(bert.transpose(1,2))
        x = ptm.ar_text_position(x)
        y_emb = ptm.ar_audio_embedding(prompt); y_pos = ptm.ar_audio_position(y_emb)
        xy_pos = torch.cat([x, y_pos], dim=1)
        x_len = x.shape[1]; y_len_init = y_emb.shape[1]; src_len = x_len + y_len_init
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len_init), value=True)
        y_attn_mask = F.pad(torch.triu(torch.ones(y_len_init, y_len_init, dtype=torch.bool), diagonal=1),
                            (x_len, 0), value=False)
        xy_attn_mask = (torch.cat([x_attn_mask_pad, y_attn_mask], dim=0).unsqueeze(0)
                        .expand(ptm.num_head, -1, -1).view(1, ptm.num_head, src_len, src_len).bool())
        xy_dec, kc, vc = ptm.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
        pt_toks = []
        for s in range(n_steps):
            logits = ptm.ar_predict_layer(xy_dec[:, -1])
            if s < 11: logits = logits[:, :-1]
            tok = int(torch.argmax(logits, dim=-1)[0]); pt_toks.append(tok)
            pos_idx = y_len_init + s
            y_emb_step = ptm.ar_audio_embedding(torch.tensor([[tok]], dtype=torch.long))
            xy_pos = (y_emb_step * ptm.ar_audio_position.x_scale
                      + ptm.ar_audio_position.alpha * ptm.ar_audio_position.pe[:, pos_idx].unsqueeze(1))
            xy_dec, kc, vc = ptm.t2s_transformer.decode_next_token(xy_pos, kc, vc)

    # MLX rollout
    phones_mx = mx.array(phones.numpy().astype(np.int32))
    bert_mx = mx.array(bert.numpy())
    prompt_mx = mx.array(prompt.numpy().astype(np.int32))
    text_emb = m.make_text_prefix(phones_mx, bert_mx)
    hidden, kcs_mx, vcs_mx, _ = m.prefill(text_emb, prompt_mx)
    y_len_init_mx = prompt_mx.shape[1]; mx_toks = []
    for s in range(n_steps):
        logits = m.logits(hidden)
        if s < 11: logits = logits[:, :-1]
        tok = int(mx.argmax(logits, axis=-1)[0]); mx_toks.append(tok)
        last_tok = mx.array([[tok]], dtype=mx.int32)
        hidden, kcs_mx, vcs_mx = m.decode_step(last_tok, y_len_init_mx + s, kcs_mx, vcs_mx)
    mx.eval(hidden)

    print(f"  PT  tokens: {pt_toks}")
    print(f"  MLX tokens: {mx_toks}")
    print(f"  match: {pt_toks == mx_toks}")
    assert pt_toks == mx_toks, f"divergence at step {next(i for i,(a,b) in enumerate(zip(pt_toks,mx_toks)) if a!=b)}"


if __name__ == "__main__":
    ptm, m = test_prefill()
    test_rollout(ptm, m, n_steps=16)
    print("\nAll tests passed.")
