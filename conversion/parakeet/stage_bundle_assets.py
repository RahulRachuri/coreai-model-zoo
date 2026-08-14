"""Stage the three non-weight files a Parakeet bundle needs beside its `.aimodel` graphs.

The graphs carry acoustics and logits; everything on either side of them is host work, and
that host work needs assets:

  tokenizer.json          the BPE + Metaspace decoder that turns emitted ids into text
  tokenizer_config.json   retagged `tokenizer_class` -> "PreTrainedTokenizer", because
                          swift-transformers rejects the upstream "ParakeetTokenizer"
                          (`unsupportedTokenizer`). Decode is driven by tokenizer.json's
                          Metaspace decoder, so the retag is exact, not an approximation.
  mel_filters_128x257_f32.bin
                          the librosa-slaney mel filterbank the Swift/Accelerate log-mel
                          front end multiplies the power spectrum by, mel-major [128, 257]
                          f32. Derived, not downloaded — it is a function of
                          (sr 16000, n_fft 512, n_mels 128, fmin 0, fmax 8000, slaney).

Both consumers read the same staged directory: `ParakeetSelfTest.swift` copies the two
tokenizer files out of `<artifacts>/bundle_assets/`, and `_parakeet_hf_upload.py` publishes
all three. Staging them once is what keeps a v2 bundle from shipping v3's tokenizer — the
vocabularies are 1025 and 8193 entries, so a mismatch is silent garbage, not an error.

    ../../.venv/bin/python stage_bundle_assets.py \
        --hf-id ../../../parakeet-v2-hf --artifacts artifacts_v2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# swift-transformers has no "ParakeetTokenizer"; this is the registered class that maps to
# BPETokenizer. See models/parakeet/README.md lesson 4.
SWIFT_TOKENIZER_CLASS = "PreTrainedTokenizer"

MEL = dict(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney")
MEL_FILE = "mel_filters_128x257_f32.bin"


def _source_file(hf_id: str, name: str) -> Path:
    """`name` from a local HF-layout directory, or from the Hub cache."""
    local = Path(hf_id).expanduser()
    if local.is_dir():
        return local / name
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(hf_id, name))


def stage(hf_id: str, dest: Path) -> Path:
    """Write the tokenizer pair + mel filterbank into `dest`. Returns `dest`."""
    import librosa

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_source_file(hf_id, "tokenizer.json"), dest / "tokenizer.json")

    cfg = json.loads(_source_file(hf_id, "tokenizer_config.json").read_text())
    cfg["tokenizer_class"] = SWIFT_TOKENIZER_CLASS
    (dest / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))

    fb = librosa.filters.mel(**MEL).astype(np.float32)  # [128, 257], mel-major
    assert fb.shape == (128, 257), fb.shape
    fb.tofile(dest / MEL_FILE)

    # Report the id space the joint head has to match. Counting entries would be wrong:
    # <unk> is an added token that *reuses* id 0, so the highest id is the only honest
    # measure (v2 -> 1025 with <blank> at 1024; v3 -> 8193 with <blank> at 8192).
    tok = json.loads((dest / "tokenizer.json").read_text())
    ids = 1 + max([*tok["model"]["vocab"].values(),
                   *(a["id"] for a in tok.get("added_tokens", []))])
    print(f"[assets] {dest}  tokenizer ids {ids}  mel {fb.shape}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default="nvidia/parakeet-tdt-0.6b-v3",
                    help="HF repo id, or a local dir in HF layout")
    ap.add_argument("--artifacts", default="artifacts",
                    help="bundle dir (relative to this script); assets land in its bundle_assets/")
    args = ap.parse_args()
    stage(args.hf_id, HERE / args.artifacts / "bundle_assets")


if __name__ == "__main__":
    main()
