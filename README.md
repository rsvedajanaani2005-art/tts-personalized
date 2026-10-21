# Personalized Text-to-Speech with Tacotron 2 and HiFi-GAN

A 500-utterance personalized text-to-speech (TTS) experiment using **Tacotron 2** for text-to-mel-spectrogram prediction and **HiFi-GAN** for waveform generation, with evaluation of the synthesized speech using ASR and speaker-embedding metrics.

## Project overview

This project explores personalized speech synthesis using a dataset of 500 recorded utterances. The TTS pipeline uses two complementary components:

1. **Tacotron 2 — acoustic model:** takes the input text and predicts a mel-spectrogram representation of the speech.
2. **HiFi-GAN — neural vocoder:** converts the predicted mel-spectrogram into an audible waveform.

Together, these components form a text-to-speech pipeline: text → mel-spectrogram → synthesized speech waveform.

The evaluation utilities in this repository assess generated audio. They are separate from the model-training pipeline: this repository does not itself train or fine-tune Tacotron 2 or HiFi-GAN.

## Repository details

- **Repository name:** `personalized-tts-500-utterance-evaluation`
- **Project title:** Personalized Text-to-Speech with Tacotron 2 and HiFi-GAN
- **Description:** Personalized TTS using Tacotron 2 and HiFi-GAN, evaluated with ASR-based WER/CER and speaker-embedding similarity.
- **Topics:** `text-to-speech`, `tacotron2`, `hifi-gan`, `speech-synthesis`, `speechbrain`, `whisper`, `audio-evaluation`

## Model architecture

```text
Input text
    │
    ▼
Tacotron 2 acoustic model
    │
    ▼
Predicted mel-spectrogram
    │
    ▼
HiFi-GAN neural vocoder
    │
    ▼
Synthesized speech waveform
```

**Tacotron 2** models the relationship between text and acoustic features. **HiFi-GAN** generates the time-domain audio waveform from the mel-spectrogram. The final quality depends on the text normalization, training data, model checkpoints, inference settings, and recording conditions.

## Evaluation metrics

The evaluation script in `src/evaluate_tts.py` compares generated WAV files with reference recordings and transcripts. It reports:

- **WER (Word Error Rate):** word recognition error in the synthesized audio; lower is better.
- **CER (Character Error Rate):** character recognition error; lower is better.
- **Reference-audio WER baseline:** ASR error on the original recording, which helps contextualize generated-audio WER.
- **Speaker embedding cosine similarity:** a rough measure of voice-identity similarity; higher is generally better.

The script uses Whisper Small for automatic speech recognition and SpeechBrain ECAPA speaker embeddings. These are proxy metrics, not a substitute for listening tests or a formal perceptual evaluation. WER/CER do not directly measure naturalness, and speaker similarity has no universal pass threshold.

## Privacy warning

Voice recordings and voice models can be sensitive personal data. Do not upload your recordings, generated samples, transcripts, checkpoints, or credentials to a public GitHub repository unless you have deliberately decided it is safe and have permission from everyone recorded. The `.gitignore` excludes common audio and model files.

## Repository structure

```text
personalized-tts-500-utterance-evaluation/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── evaluate_tts.py
└── data/
    └── .gitkeep
```

## Expected local data layout

Keep the audio and metadata outside the public repository:

```text
MyTTSData/
├── reference_audio/       # original recordings: 001.wav ... 500.wav
├── generated_audio/       # synthesized WAV files with matching IDs
└── metadata_001-500.csv   # pipe-delimited metadata
```

The metadata file should have no header and contain either two columns (`ID|transcript`) or three (`ID|transcript|normalized_transcript`). Example:

```text
001|The first sentence.|The first sentence.
002|The second sentence.|The second sentence.
```

The script matches numeric IDs in the metadata and WAV filenames. For example, `001.wav` matches metadata ID `001`.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The first evaluation run downloads Whisper and SpeechBrain model weights. A GPU is helpful but not mandatory.

## Run evaluation

```bash
python src/evaluate_tts.py --reference-dir "/path/to/MyTTSData/reference_audio" --generated-dir "/path/to/MyTTSData/generated_audio" --metadata "/path/to/MyTTSData/metadata_001-500.csv" --output "results/tts_evaluation_results.csv"
```

The `generated_audio` directory should contain WAV files synthesized by your Tacotron 2 + HiFi-GAN pipeline. Ensure the filenames correspond to the reference utterance IDs.

## Interpreting results

- Lower generated WER/CER generally means the generated audio is easier for the selected ASR model to transcribe correctly.
- Reference-audio WER is a useful baseline. If Whisper struggles on the original recording, generated-audio WER may also reflect ASR limitations.
- Higher speaker cosine similarity generally indicates more similar embeddings, but it does not guarantee that listeners will perceive the voices as identical.
- Listen to a consistent set of samples and check pronunciation, pauses, prosody, clipping, background noise, and voice resemblance.
- To compare experiments using 100, 200, 300, and 500 utterances fairly, evaluate the same held-out sentences for each model. Ideally, held-out sentences should not have been used during training.

## Limitations

1. This repository contains evaluation code, not the Tacotron 2/HiFi-GAN training implementation.
2. WER/CER evaluate transcript agreement, not speech naturalness.
3. Speaker embeddings can be affected by audio duration, noise, loudness, and recording conditions.
4. The script expects WAV files whose filenames contain matching numeric IDs.
5. Add measured results only after running the evaluation; do not report unmeasured metrics.

## Suggested next steps

1. Generate audio with your Tacotron 2 + HiFi-GAN pipeline.
2. Evaluate generated audio on held-out utterances.
3. Listen to a fixed set of reference/generated pairs.
4. Compare results from different training-data sizes in a results table.
5. Document training settings and checkpoints separately, without publishing private voice data.

## License

Choose a license after deciding what others may reuse. A code license does not grant permission to use private recordings or model weights.
