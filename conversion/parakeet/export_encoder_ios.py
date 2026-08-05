"""Phase 2b — re-author the Parakeet FastConformer encoder on the **iOS/ANE authoring track**
and export it to a static Core AI .aimodel, gated against the same golden `enc_proj`.

Why a second author: `export_encoder.py` writes the GPU/macOS track (standard `(B,T,D)` layout,
`nn.Linear`, batched multi-head SDPA, fp32 intermediates). That graph runs beautifully on the
Metal delegate but only ~25 fragments of it are ANE-eligible, so `preferredComputeUnitKind:
.neuralEngine` ends up *slower* (the rest bounces back to the GPU). The ANE track is a purely
mechanical re-authoring of the SAME fp16 weights — no quantization, no approximation:

  * **BC1S** — activations live as `(B, S, 1, C)` between blocks and are transposed to
    `(B, C, 1, S)` around every projection (the shape `primitives/ios/*` uses).
  * **1×1 Conv2d instead of nn.Linear** — Conv2d maps to the ANE convolution engine (which also
    accumulates in fp32); Linear decomposes into ops that fall off-ANE.
  * **Per-head sequential attention** via the `bchq,bkhc->bkhq` einsum — the ANE has no fused SDPA.
  * **Mel image transposed so TIME is the width axis** — the ANE pads the last axis to 64 B, and
    the GPU track's freq-last subsampling ends at width 16 (32 B, i.e. 2× waste). Time-last ends at
    width 361. The 3×3 kernels are transposed to match; the maths is identical.
  * **No fp32 literals** — the two Python scalars in the reference implementation (the 0.5 on each
    half-FFN and the 1/sqrt(d) attention scale) are FOLDED INTO WEIGHTS at build time, so the graph
    carries none. BatchNorm is likewise folded into the depthwise conv (exact at inference).
  * **rel-pos keys baked** — `relative_k_proj(pos_emb)` depends only on constants, so it is
    evaluated at build time and stored as a per-layer fp16 buffer (removes 24 convs).

Everything else — weights, block order, rel-shift semantics — is bit-identical in intent to
`export_encoder.py`. Full precision throughout: fp16 encoder, exactly like the shipping one.

Run (MAIN venv):
    .venv/bin/python export_encoder_ios.py --hf-id ~/Developer/Personal/parakeet-v2-hf \
        --oracle oracle_v2_30s.npz --artifacts artifacts_v2_ios
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEFAULT_HF_ID = "nvidia/parakeet-tdt-0.6b-v3"
HID, HEADS, HDIM, FF, NLAYERS = 1024, 8, 128, 4096, 24
MEL, SUB_CH, VOCAB_PROJ = 128, 256, 640
CONV_K = 9
BN_EPS = 1e-5


# --------------------------------------------------------------------------- helpers
def build_pos_emb(T: int) -> torch.Tensor:
    """Transformer-XL sinusoid table, [1, 2T-1, HID]. Identical to the GPU track."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HID, 2, dtype=torch.float) / HID))
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float)            # [2T-1]
    freqs = (inv_freq[:, None] @ pos[None, :]).transpose(0, 1)      # [2T-1, HID/2]
    pe = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(2 * T - 1, HID)
    return pe[None]


def subsampled_len(L: int) -> int:
    T = L
    for _ in range(3):
        T = (T + 2 - 3) // 2 + 1
    return T


# --------------------------------------------------------------------------- modules
class FeedForwardANE(nn.Module):
    """½·FFN. The ½ is folded into linear2's weights, so no fp32 literal reaches the graph.

    Two shapes for the same maths, selected by ``--ffn``:
      * ``width``  — BC1S `(B,HID,1,S)`, sequence on the conv's width axis. Needs a real transpose
        in and out, because `(B,S,1,C)` -> `(B,C,1,S)` moves the innermost axis.
      * ``batch``  — the shipped `primitives/ios/mlp.py` form: `(B·S, HID, 1, 1)`. From `(B,S,1,C)`
        that is a pure VIEW (C is already innermost), so both reshapes are free, and the sequence
        becomes conv batch instead of conv width.
    """

    MODE = "width"

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Conv2d(HID, FF, 1, bias=False)
        self.linear2 = nn.Conv2d(FF, HID, 1, bias=False)

    def forward(self, x):                      # x [B,S,1,HID]
        B, S = x.shape[0], x.shape[1]
        if self.MODE == "batch":
            x = x.reshape(B * S, HID, 1, 1)    # free: HID is already the innermost axis
            x = self.linear2(F.silu(self.linear1(x)))
            return x.reshape(B, S, 1, HID)
        x = x.transpose(-3, -1)                # -> [B,HID,1,S]
        x = self.linear2(F.silu(self.linear1(x)))
        return x.transpose(-3, -1)             # -> [B,S,1,HID]


class ConvModuleANE(nn.Module):
    """Conformer conv module in BC1S. Conv1d(k) over time becomes Conv2d((1,k)) over width, and
    the BatchNorm is folded into the depthwise conv's weights + a bias (exact at inference)."""

    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv2d(HID, 2 * HID, 1, bias=False)
        self.depthwise_conv = nn.Conv2d(HID, HID, (1, CONV_K),
                                        padding=(0, (CONV_K - 1) // 2), groups=HID, bias=True)
        self.pointwise_conv2 = nn.Conv2d(HID, HID, 1, bias=False)

    def forward(self, x):                      # x [B,S,1,HID]
        x = x.transpose(-3, -1)                # -> [B,HID,1,S]
        x = self.pointwise_conv1(x)            # [B,2*HID,1,S]
        x = F.glu(x, dim=1)                    # gate on the CHANNEL axis -> [B,HID,1,S]
        x = self.depthwise_conv(x)             # BN folded in
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(-3, -1)             # -> [B,S,1,HID]


class RelPosAttentionANE(nn.Module):
    """Transformer-XL relative attention, per-head and sequential.

    The 1/sqrt(d) scale is folded into k_proj's weights and into the baked rel-key buffer, so the
    graph has no scalar multiply. `rel_k` (= relative_k_proj(pos_emb)) is input-independent and is
    therefore materialised at build time as a BC1S fp16 buffer.
    """

    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.q_proj = nn.Conv2d(HID, HID, 1, bias=False)
        self.k_proj = nn.Conv2d(HID, HID, 1, bias=False)   # weights carry the 1/sqrt(d) scale
        self.v_proj = nn.Conv2d(HID, HID, 1, bias=False)
        self.o_proj = nn.Conv2d(HID, HID, 1, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(1, HID, 1, 1))
        self.bias_v = nn.Parameter(torch.zeros(1, HID, 1, 1))
        # [1, HID, 1, 2T-1], pre-scaled. Filled by load_weights().
        self.register_buffer("rel_k", torch.zeros(1, HID, 1, 2 * T - 1), persistent=False)

    def _rel_shift_qmajor(self, x):
        """x [B, Q, 1, 2T-1] (query-major, relative-offset on the WIDTH axis) -> [B, K, 1, Q].

        Same pad/reshape identity the GPU track uses, but with the offset axis innermost — that is
        the only orientation in which the trick is expressible as reshapes (the per-row constant
        offset has to run along the fast axis). Net effect: out[k, q] = x[q, T-1-q+k].
        """
        B, Q, _, P = x.shape
        T = self.T
        x = x.reshape(B, Q, P)
        x = torch.cat([x.new_zeros(B, Q, 1), x], dim=-1)   # [B,Q,2T]
        x = x.reshape(B, 2 * T, T)
        x = x[:, 1:, :]                                    # [B,2T-1,T]
        x = x.reshape(B, T, P)
        x = x[:, :, :T]                                    # [B,Q,K]
        return x.permute(0, 2, 1).unsqueeze(2)             # [B,K,1,Q]

    def forward(self, x):                                  # x [B,HID,1,S] (BC1S)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q_u = q + self.bias_u
        q_v = q + self.bias_v

        scores = []
        for h in range(HEADS):
            sl = slice(h * HDIM, (h + 1) * HDIM)
            # "bchq,bkhc->bkhq" — the documented ANE attention einsum: no reshape copies, and the
            # result is KEY-major so the softmax lands on the channel axis.
            ac = torch.einsum("bchq,bkhc->bkhq",
                              q_u[:, sl], k[:, sl].transpose(-3, -1))      # [B,K,1,Q]
            # Same einsum with the roles swapped gives the rel term QUERY-major, which is what
            # _rel_shift_qmajor needs.
            bd = torch.einsum("bchq,bkhc->bkhq",
                              self.rel_k[:, sl], q_v[:, sl].transpose(-3, -1))  # [B,Q,1,P]
            scores.append(ac + self._rel_shift_qmajor(bd))

        attn = torch.softmax(torch.cat(scores, dim=2), dim=1)              # [B,K,HEADS,Q]

        outs = []
        for h in range(HEADS):
            sl = slice(h * HDIM, (h + 1) * HDIM)
            outs.append(torch.einsum("bkhq,bkhc->bchq",
                                     attn[:, :, h:h + 1], v[:, sl].transpose(-3, -1)))
        return self.o_proj(torch.cat(outs, dim=1))                         # [B,HID,1,S]


class ConformerBlockANE(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.feed_forward1 = FeedForwardANE()
        self.self_attn = RelPosAttentionANE(T)
        self.conv = ConvModuleANE()
        self.feed_forward2 = FeedForwardANE()
        self.norm_feed_forward1 = nn.LayerNorm(HID)
        self.norm_self_att = nn.LayerNorm(HID)
        self.norm_conv = nn.LayerNorm(HID)
        self.norm_feed_forward2 = nn.LayerNorm(HID)
        self.norm_out = nn.LayerNorm(HID)

    def forward(self, x):                                  # x [B,S,1,HID]
        x = x + self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x).transpose(-3, -1)).transpose(-3, -1)
        x = x + self.conv(self.norm_conv(x))
        x = x + self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class SubsamplingANE(nn.Module):
    """8× conv subsampling on the mel image with TIME on the width axis.

    The GPU track feeds `(B,1,L,MEL)` (freq = width). Same convolution, transposed image: we feed
    `(B,1,MEL,L)` and transpose each 3×3 kernel, which by symmetry of stride/padding produces the
    transposed output. Width then runs 2885→1443→722→361 instead of 128→64→32→16.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, SUB_CH, 3, stride=2, padding=1),                       # 0
            nn.ReLU(),                                                          # 1
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),   # 2
            nn.Conv2d(SUB_CH, SUB_CH, 1),                                       # 3
            nn.ReLU(),                                                          # 4
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),   # 5
            nn.Conv2d(SUB_CH, SUB_CH, 1),                                       # 6
            nn.ReLU(),                                                          # 7
        ])
        self.linear = nn.Conv2d(SUB_CH * (MEL // 8), HID, 1, bias=True)

    def forward(self, mel):                    # mel [B,MEL,L]
        x = mel.unsqueeze(1)                   # [B,1,MEL,L]  (freq=H, time=W)
        for layer in self.layers:
            x = layer(x)
        B, C, Fq, T = x.shape                  # [B,256,16,T]
        # Merging (C,Fq) row-major gives feature index c*16+f — the same ordering the GPU track's
        # transpose+reshape produces, so `linear`'s weights need no permutation.
        x = x.reshape(B, C * Fq, 1, T)         # [B,4096,1,T]
        return self.linear(x).transpose(-3, -1)  # [B,T,1,HID]


class EncoderANE(nn.Module):
    def __init__(self, mel_len: int):
        super().__init__()
        self.subsampling = SubsamplingANE()
        self.T = T = subsampled_len(mel_len)
        self.layers = nn.ModuleList([ConformerBlockANE(T) for _ in range(NLAYERS)])
        self.projector = nn.Conv2d(HID, VOCAB_PROJ, 1, bias=True)

    def forward(self, mel):                    # mel [1,MEL,L]
        x = self.subsampling(mel)              # [B,T,1,HID]
        for blk in self.layers:
            x = blk(x)
        x = self.projector(x.transpose(-3, -1))          # [B,640,1,T]
        B, C, _, T = x.shape
        return x.transpose(-3, -1).reshape(B, T, C)      # [B,T,640]


# --------------------------------------------------------------------------- weights
def weights_file(hf_id: str) -> str:
    local = Path(hf_id) / "model.safetensors"
    if local.is_file():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(hf_id, "model.safetensors")


def load_weights(enc: EncoderANE, hf_id: str):
    """Map the HF checkpoint onto the ANE authoring, applying the mechanical rewrites."""
    from safetensors import safe_open

    src = {}
    with safe_open(weights_file(hf_id), framework="pt") as f:
        for key in f.keys():
            src[key] = f.get_tensor(key).float()

    def take(name):
        return src.pop(name)

    sd = {}
    scale = HDIM ** -0.5

    # ---- subsampling: transpose the 3×3 kernels (image is transposed), Linear -> 1×1 Conv2d
    for idx in (0, 2, 3, 5, 6):
        w = take(f"encoder.subsampling.layers.{idx}.weight")
        if w.shape[-1] == 3 and w.shape[-2] == 3:
            w = w.transpose(-1, -2).contiguous()
        sd[f"subsampling.layers.{idx}.weight"] = w
        sd[f"subsampling.layers.{idx}.bias"] = take(f"encoder.subsampling.layers.{idx}.bias")
    sd["subsampling.linear.weight"] = take("encoder.subsampling.linear.weight")[:, :, None, None]
    sd["subsampling.linear.bias"] = take("encoder.subsampling.linear.bias")

    # ---- rel-pos keys: constant-folded per layer (they only depend on the sinusoid table)
    pos_emb = build_pos_emb(enc.T)                                  # [1,2T-1,HID]

    for i in range(NLAYERS):
        p = f"encoder.layers.{i}."
        t = f"layers.{i}."

        for fk, half in (("feed_forward1", True), ("feed_forward2", True)):
            sd[t + fk + ".linear1.weight"] = take(p + fk + ".linear1.weight")[:, :, None, None]
            w2 = take(p + fk + ".linear2.weight")
            if half:
                w2 = w2 * 0.5          # the residual's ½ folded in — no fp32 literal in the graph
            sd[t + fk + ".linear2.weight"] = w2[:, :, None, None]

        for proj, mul in (("q_proj", 1.0), ("k_proj", scale), ("v_proj", 1.0), ("o_proj", 1.0)):
            w = take(p + "self_attn." + proj + ".weight") * mul
            sd[t + "self_attn." + proj + ".weight"] = w[:, :, None, None]

        sd[t + "self_attn.bias_u"] = take(p + "self_attn.bias_u").reshape(1, HID, 1, 1)
        sd[t + "self_attn.bias_v"] = take(p + "self_attn.bias_v").reshape(1, HID, 1, 1)

        rel_w = take(p + "self_attn.relative_k_proj.weight")         # [HID,HID]
        rel_k = (pos_emb @ rel_w.t()) * scale                        # [1,2T-1,HID]
        enc.layers[i].self_attn.rel_k.copy_(rel_k.permute(0, 2, 1).unsqueeze(2))

        # ---- conv module: Conv1d -> Conv2d((1,k)); BatchNorm folded into the depthwise conv
        sd[t + "conv.pointwise_conv1.weight"] = take(p + "conv.pointwise_conv1.weight").unsqueeze(2)
        sd[t + "conv.pointwise_conv2.weight"] = take(p + "conv.pointwise_conv2.weight").unsqueeze(2)

        dw = take(p + "conv.depthwise_conv.weight")                  # [HID,1,K]
        gamma = take(p + "conv.norm.weight")
        beta = take(p + "conv.norm.bias")
        mean = take(p + "conv.norm.running_mean")
        var = take(p + "conv.norm.running_var")
        src.pop(p + "conv.norm.num_batches_tracked", None)
        alpha = gamma / torch.sqrt(var + BN_EPS)                     # per-channel
        sd[t + "conv.depthwise_conv.weight"] = (dw * alpha[:, None, None]).unsqueeze(2)
        sd[t + "conv.depthwise_conv.bias"] = beta - mean * alpha

        for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv",
                     "norm_feed_forward2", "norm_out"):
            sd[t + norm + ".weight"] = take(p + norm + ".weight")
            sd[t + norm + ".bias"] = take(p + norm + ".bias")

    sd["projector.weight"] = take("encoder_projector.weight")[:, :, None, None]
    sd["projector.bias"] = take("encoder_projector.bias")

    missing, unexpected = enc.load_state_dict(sd, strict=False)
    miss = [m for m in missing if not m.endswith("rel_k")]
    assert not miss and not unexpected, f"weight map mismatch: missing={miss[:5]} unexpected={unexpected[:5]}"
    print(f"[weights] loaded {len(sd)} tensors + {NLAYERS} baked rel-key buffers (strict map OK)")


# --------------------------------------------------------------------------- self-test
def check_rel_shift(T: int):
    """The ANE rel-shift must agree with the GPU track's pad/view identity on random data."""
    def gpu_rel_shift(x):                      # x [B,H,T,2T-1] -> [B,H,T,2T-1]
        b, h, t, p = x.shape
        x = F.pad(x, (1, 0)).view(b, h, p + 1, t)
        return x[:, :, 1:].reshape(b, h, t, p)

    P = 2 * T - 1
    ref_in = torch.randn(1, 1, T, P)
    want = gpu_rel_shift(ref_in)[..., :T][0, 0]                       # [Q,K]

    att = RelPosAttentionANE(T)
    got = att._rel_shift_qmajor(ref_in.reshape(1, T, 1, P))[0, :, 0, :]  # [K,Q]
    assert torch.equal(want, got.t()), "rel-shift mismatch between GPU and ANE formulations"
    print(f"[self-test] rel-shift identity holds for T={T}")


# --------------------------------------------------------------------------- gate/export
def build_program(enc, example, decomp: str):
    """torch.export -> Core AI, replicating `export/ios.py`'s decomposition handling.

    `export/ios.py` itself is LLM-shaped (4 KV-cache entrypoints), so it is not callable here —
    what transfers is its decomp policy: the DEFAULT aten table minus SiLU, so SiLU survives as one
    op instead of being split into sigmoid+mul. `--decomp macos` selects the GPU track's
    `coreai_torch.get_decomp_table()` for comparison.
    """
    import coreai_torch
    from coreai_torch import TorchConverter
    from coreai_models.export.mlir_ops import (
        register_custom_torch_lowering,
        remove_functionalization,
    )

    with torch.no_grad():
        ep = torch.export.export(enc, args=(), kwargs=example, dynamic_shapes=None)
        if decomp == "ios":
            table = torch.export.default_decompositions()
            table.pop(torch.ops.aten.silu.default, None)
            table.pop(torch.ops.aten.silu.out, None)
        else:
            table = coreai_torch.get_decomp_table()
        ep = ep.run_decompositions(table)
    remove_functionalization(ep)

    converter = TorchConverter()
    register_custom_torch_lowering(converter)
    converter.add_exported_program(
        ep, input_names=["mel"], output_names=["enc_proj"], entrypoint_name="main")
    return converter.to_coreai()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID, help="HF repo id, or a local dir in HF layout")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--oracle", default="oracle_v2_30s.npz")
    ap.add_argument("--artifacts", default="artifacts_v2_ios", help="output dir (rel. to script)")
    ap.add_argument("--decomp", choices=["ios", "macos"], default="ios")
    ap.add_argument("--ffn", choices=["width", "batch"], default="width",
                    help="FFN conv geometry: sequence on the width axis, or on the batch axis "
                         "(the shipped primitives/ios/mlp.py form, which reshapes for free)")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--gate-units", default="ane,gpu", help="comma list, or empty to skip")
    args = ap.parse_args()

    d = np.load(HERE / args.oracle)
    mel = torch.from_numpy(d["input_features"]).float()    # [1,L,128] (processor layout)
    golden = torch.from_numpy(d["enc_proj"]).float()       # [T,640]
    L = mel.shape[1]
    mel_in = mel.transpose(1, 2).contiguous()              # [1,128,L]
    print(f"[model] {args.hf_id}  (iOS/ANE authoring track)")
    print(f"mel {tuple(mel_in.shape)} L={L} golden enc_proj {tuple(golden.shape)}")
    src = str(d["source"]) if "source" in d else None
    if src and src != args.hf_id:
        print(f"⚠️  {args.oracle} was generated from {src!r}, not {args.hf_id!r}")

    check_rel_shift(subsampled_len(L))

    FeedForwardANE.MODE = args.ffn
    enc = EncoderANE(L).eval()
    load_weights(enc, args.hf_id)
    assert enc.T == golden.shape[0], f"T mismatch {enc.T} vs {golden.shape[0]}"

    with torch.no_grad():
        out = enc(mel_in)[0]                                # [T,640]
    cos = torch.nn.functional.cosine_similarity(out.reshape(-1), golden.reshape(-1), dim=0).item()
    pertok = torch.nn.functional.cosine_similarity(out, golden, dim=-1)
    print(f"[eager fp32] global cos {cos:.6f}  per-token mean {pertok.mean():.6f} "
          f"min {pertok.min():.6f}  max|Δ| {(out - golden).abs().max():.4f}")
    if pertok.mean() < 0.999:
        print("❌ ANE re-author DIVERGES — fix before export")
        raise SystemExit(1)
    print("✅ eager ANE re-author matches golden")
    if args.skip_export:
        return

    import asyncio
    import coreai.runtime as rt

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    enc_d = enc.to(dtype)
    example = {"mel": torch.zeros(1, MEL, L, dtype=dtype)}
    print(f"[export] ANE-track FastConformer encoder ({args.dtype}, decomp={args.decomp}) ...",
          flush=True)
    prog = build_program(enc_d, example, args.decomp)
    prog.optimize()
    art = HERE / args.artifacts
    art.mkdir(exist_ok=True)
    aimodel = art / f"parakeet_encoder_{args.dtype}_L{L}.aimodel"
    shutil.rmtree(aimodel, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "cc-by-4.0"
    prog.save_asset(aimodel, meta)
    sz = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {aimodel} ({sz:.1f} MB)")

    async def gate(unit):
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(
                    getattr(rt.ComputeUnitKind, {"ane": "neural_engine"}.get(unit, unit))()))
        m = await rt.AIModel.load(str(aimodel), opts)
        fn = m.load_function("main")
        res = await fn({"mel": rt.NDArray(mel_in.to(dtype).numpy())})
        eng = torch.from_numpy(res["enc_proj"].numpy().astype(np.float32))[0]
        pt = torch.nn.functional.cosine_similarity(eng, golden, dim=-1)
        ok = pt.mean() > 0.999 and pt.min() > 0.99
        print(f"[gate {unit}] per-token cos mean {pt.mean():.6f} min {pt.min():.6f} "
              f"max|Δ| {(eng - golden).abs().max():.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    for unit in [u for u in args.gate_units.split(",") if u]:
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
