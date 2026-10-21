#!/usr/bin/env python3
"""Evaluate generated TTS WAV files against references and a pipe-delimited transcript file."""
import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from jiwer import wer, cer
from transformers import pipeline
from speechbrain.inference.speaker import EncoderClassifier


def normalize_id(value):
    match = re.search(r"\d+", str(value))
    return str(int(match.group())) if match else str(value).strip()


def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def index_wavs(folder):
    indexed = {}
    for path in Path(folder).rglob("*.wav"):
        key = normalize_id(path.stem)
        if key in indexed:
            print(f"WARNING: duplicate ID {key}; keeping {indexed[key]}, skipping {path}")
            continue
        indexed[key] = str(path)
    return indexed


def load_metadata(path):
    df = pd.read_csv(path, sep="|", header=None, dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError("Metadata needs at least ID and transcript columns.")
    if df.shape[1] >= 3:
        df = df.iloc[:, :3]
        df.columns = ["id", "transcript", "normalized_transcript"]
    else:
        df = df.iloc[:, :2]
        df.columns = ["id", "transcript"]
    mapping = {}
    for _, row in df.iterrows():
        key = normalize_id(row["id"])
        normalized = row.get("normalized_transcript", "")
        mapping[key] = normalized.strip() if normalized.strip() else row["transcript"].strip()
    return mapping


def load_audio(path):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="results/tts_evaluation_results.csv")
    parser.add_argument("--asr-model", default="openai/whisper-small")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    texts = load_metadata(args.metadata)
    references = index_wavs(args.reference_dir)
    generated = index_wavs(args.generated_dir)
    ids = sorted(set(texts) & set(references) & set(generated),
                 key=lambda x: int(x) if x.isdigit() else x)
    print(f"Metadata: {len(texts)} | reference WAVs: {len(references)} | generated WAVs: {len(generated)}")
    print(f"Matched utterances: {len(ids)}")
    if not ids:
        raise ValueError("No matching IDs. Check paths, WAV filenames, and metadata IDs.")

    asr = pipeline("automatic-speech-recognition", model=args.asr_model,
                   device=0 if device == "cuda" else -1)
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

    def transcribe(path):
        audio = load_audio(path)
        return asr({"raw": audio, "sampling_rate": 16000},
                   generate_kwargs={"task": "transcribe"})["text"].strip()

    def embed(path):
        audio = load_audio(path)
        if len(audio) < 1600:
            return None
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = speaker_model.encode_batch(waveform)
        return vector.reshape(-1).detach().cpu().numpy()

    def cosine(a, b):
        if a is None or b is None:
            return np.nan
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else np.nan

    rows = []
    for i, key in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {key}")
        ref_path, gen_path, target = references[key], generated[key], texts[key]
        try:
            ref_text, gen_text = transcribe(ref_path), transcribe(gen_path)
            sim = cosine(embed(ref_path), embed(gen_path))
            target_clean = clean_text(target)
            rows.append({
                "id": key, "reference_path": ref_path, "generated_path": gen_path,
                "target_transcript": target, "reference_asr": ref_text, "generated_asr": gen_text,
                "generated_wer": wer(target_clean, clean_text(gen_text)),
                "generated_cer": cer(target_clean, clean_text(gen_text)),
                "reference_asr_wer_baseline": wer(target_clean, clean_text(ref_text)),
                "speaker_cosine_similarity": sim, "error": ""
            })
        except Exception as exc:
            print(f"Error for {key}: {exc}")
            rows.append({
                "id": key, "reference_path": ref_path, "generated_path": gen_path,
                "target_transcript": target, "reference_asr": "", "generated_asr": "",
                "generated_wer": np.nan, "generated_cer": np.nan,
                "reference_asr_wer_baseline": np.nan,
                "speaker_cosine_similarity": np.nan, "error": str(exc)
            })

    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"\nSaved results: {output.resolve()}")
    print("\nMean metrics:")
    for column in ["generated_wer", "generated_cer",
                   "reference_asr_wer_baseline", "speaker_cosine_similarity"]:
        print(f"{column}: {result[column].mean(skipna=True):.4f}")
    print("Reminder: compare models on the same held-out utterances and listen to samples.")


if __name__ == "__main__":
    main()
