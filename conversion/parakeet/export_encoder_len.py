"""Export the Parakeet FastConformer encoder at an **arbitrary static mel length L**, and
time it — the instrument for the "matched bucket" question (Round 7).

Why this exists alongside `export_encoder.py`: that script derives L from the oracle npz
(`input_features` is [1,L,128]) and gates the fp16 export against the HF golden `enc_proj`,
which pins it to whatever L the oracle was generated at (2885). To answer "what does the
encoder cost at a *different* L" there is no golden at that L and generating one needs the
isolated transformers-5.x env.

The gate here is therefore re-anchored, not weakened: `export_encoder.py` already proved the
**eager fp32 re-author** reproduces the HF golden at L=2885 (per-token cos > 0.999). The
re-author is length-generic — only the positional table and the tensor shapes depend on L —
so at a new L the eager fp32 re-author *is* the fp32 oracle, and what needs gating is the
Core AI fp16 export against it. That is exactly the "encoder cosine vs fp32 oracle" gate,
with the oracle computed locally instead of loaded from disk.

The gate input is a **real** mel from the book (not noise): fp16 error is input-dependent, so
gating on random input would flatter the export.

Usage (zoo venv):
    .venv/bin/python conversion/parakeet/export_encoder_len.py \
        --mel-len 2880 --hf-id ~/Developer/Personal/parakeet-v2-hf \
        --artifacts artifacts_v2_matched --gate
    .venv/bin/python conversion/parakeet/export_encoder_len.py \
        --mel-len 2880 --artifacts artifacts_v2_matched --timing-only --passes 30
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_encoder import MEL, Encoder, load_weights  # noqa: E402

BENCH = Path.home() / "Developer/Personal/parakeet-bench"
DEFAULT_WAV = BENCH / "work/107days_ch1/audio_16k.wav"
DEFAULT_HF = str(Path.home() / "Developer/Personal/parakeet-v2-hf")


def real_mel(mel_len: int, wav_path: Path, chunk_index: int) -> np.ndarray:
    """A real book chunk -> [1,128,mel_len], using the *validated* Python frontend with its
    bucket constant retargeted to `mel_len`.

    Retargeting BUCKET is the whole point: the per-mel-bin normalisation runs over the entire
    padded window, so the bucket length is baked into the feature values. A mel produced at
    2885 and truncated to 2880 is NOT the mel the L=2880 pipeline would see.
    """
    sys.path.insert(0, str(BENCH))
    from bench import frontend  # noqa: E402
    import json
    import soundfile as sf

    frontend.BUCKET = mel_len
    frontend.BUCKET_SAMPLES = mel_len * frontend.HOP

    manifest = json.loads((wav_path.parent / "chunks.json").read_text())
    c = manifest["chunks"][chunk_index]
    wav, sr = sf.read(str(wav_path), start=c["start"], stop=c["end"], dtype="float32")
    assert sr == 16000, sr
    print(f"[mel] chunk {chunk_index}: {c['end'] - c['start']} samples "
          f"({c['seconds']:.2f} s) -> bucket {mel_len} frames "
          f"({mel_len * 160 / 16000:.2f} s), pad {mel_len * 160 - (c['end'] - c['start'])} samples")
    return frontend.mel_bucket(wav)


def synthetic_mel(mel_len: int) -> np.ndarray:
    """Shape-correct noise. GPU time is data-independent, so this is fine for timing only."""
    g = np.random.default_rng(0)
    return g.standard_normal((1, MEL, mel_len)).astype(np.float32)


def asset_path(artifacts: Path, mel_len: int) -> Path:
    return artifacts / f"parakeet_encoder_float16_L{mel_len}.aimodel"


def do_export(mel_len: int, hf_id: str, artifacts: Path, mel: np.ndarray | None):
    enc = Encoder(mel_len).eval()
    load_weights(enc, hf_id)
    print(f"[shape] L={mel_len} -> encoder frames T={enc.T}")

    ref = None
    if mel is not None:
        with torch.no_grad():
            ref = enc(torch.from_numpy(mel).float())[0]  # [T,640] fp32 oracle at this L
        print(f"[oracle] eager fp32 enc_proj {tuple(ref.shape)} "
              f"absmax {ref.abs().max():.4f} mean {ref.mean():.6f}")

    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    enc_d = enc.to(torch.float16)
    example = {"mel": torch.zeros(1, MEL, mel_len, dtype=torch.float16)}
    print(f"[export] fp16 FastConformer encoder L={mel_len} -> Core AI ...", flush=True)
    t0 = time.time()
    prog = export_to_coreai(enc_d, example, dynamic_shapes=None,
                            input_names=("mel",), output_names=("enc_proj",),
                            state_names=None, externalize_modules=[])
    prog.optimize()
    artifacts.mkdir(parents=True, exist_ok=True)
    out = asset_path(artifacts, mel_len)
    shutil.rmtree(out, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "cc-by-4.0"
    prog.save_asset(out, meta)
    sz = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {out} ({sz:.1f} MB) in {time.time() - t0:.1f} s")
    return ref


def gate(mel_len: int, artifacts: Path, mel: np.ndarray, ref: torch.Tensor):
    """Cosine of the GPU fp16 export vs the eager fp32 re-author at the SAME L."""
    import asyncio
    import coreai.runtime as rt

    async def run():
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        m = await rt.AIModel.load(str(asset_path(artifacts, mel_len)), opts)
        fn = m.load_function("main")
        res = await fn({"mel": rt.NDArray(mel.astype(np.float16))})
        return torch.from_numpy(res["enc_proj"].numpy().astype(np.float32))[0]

    got = asyncio.run(run())
    pt = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    glob = torch.nn.functional.cosine_similarity(got.reshape(-1), ref.reshape(-1), dim=0)
    ok = pt.mean() > 0.999 and pt.min() > 0.99
    print(f"[gate gpu L={mel_len}] global cos {glob:.6f}  per-token cos mean {pt.mean():.6f} "
          f"min {pt.min():.6f}  max|delta| {(got - ref).abs().max():.4f} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return bool(ok)


def timing(mel_len: int, artifacts: Path, passes: int, warmup: int):
    """Warm, serial `run()` timing of one encoder pass. Rule 7: keep total calls per process
    well under ~5k (IOSurface leak)."""
    import asyncio
    import coreai.runtime as rt

    mel = synthetic_mel(mel_len).astype(np.float16)

    async def run():
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        t0 = time.time()
        m = await rt.AIModel.load(str(asset_path(artifacts, mel_len)), opts)
        fn = m.load_function("main")
        load_s = time.time() - t0
        arr = rt.NDArray(mel)
        for _ in range(warmup):
            await fn({"mel": arr})
        ts = []
        for _ in range(passes):
            t = time.time()
            await fn({"mel": arr})
            ts.append(time.time() - t)
        return load_s, ts

    load_s, ts = asyncio.run(run())
    ts_ms = sorted(t * 1000 for t in ts)
    med = statistics.median(ts_ms)
    print(f"[time L={mel_len}] load(+JIT) {load_s:.1f} s | n={passes} "
          f"median {med:.2f} ms  min {ts_ms[0]:.2f}  p90 {ts_ms[int(.9 * len(ts_ms))]:.2f} "
          f"max {ts_ms[-1]:.2f}  | per-mel-frame {med / mel_len * 1000:.3f} us")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel-len", type=int, required=True)
    ap.add_argument("--hf-id", default=DEFAULT_HF)
    ap.add_argument("--artifacts", required=True, help="output dir (relative to this script)")
    ap.add_argument("--wav", default=str(DEFAULT_WAV))
    ap.add_argument("--chunk", type=int, default=0, help="manifest chunk index for the gate mel")
    ap.add_argument("--gate", action="store_true", help="export + cosine gate on a real mel")
    ap.add_argument("--timing-only", action="store_true", help="skip export, time an existing asset")
    ap.add_argument("--export-only", action="store_true", help="export with no gate (timing assets)")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    artifacts = HERE / args.artifacts
    if args.timing_only:
        timing(args.mel_len, artifacts, args.passes, args.warmup)
        return

    mel = None if args.export_only else real_mel(args.mel_len, Path(args.wav), args.chunk)
    ref = do_export(args.mel_len, args.hf_id, artifacts, mel)
    if args.gate:
        ok = gate(args.mel_len, artifacts, mel, ref)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
