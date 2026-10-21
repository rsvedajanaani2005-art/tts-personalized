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


## Experimental Settings

### HiFi-GAN Vocoder Configuration

HiFi-GAN was used as the neural vocoder to convert the mel-spectrograms predicted by Tacotron 2 into synthesized speech waveforms. The vocoder training configuration is summarized below.

| Parameter | Value |
|---|---|
| Vocoder | HiFi-GAN |
| Training Epochs | 15 |
| Batch Size | 4 |
| Learning Rate | 0.00005 |
| Optimizer | Ada, optimizer |
| Sampling Rate | 22,050 Hz |
| Training Dataset | 500 utterances |
| Checkpoint Selection | Train loss |

### Dataset Split

The dataset containing 500 recorded utterances was divided into training, validation, and test subsets.

| Subset | Number of Utterances | 
|---|---:|
| Training | 450 | 
| Validation | 50 | 
| **Total** | **500** | 

The training subset was used to optimize the model parameters. The validation subset was used to monitor training and support model selection, while the test subset was reserved for final evaluation.

### Training Details

- **Input representation:** Mel-spectrograms generated from speech recordings.
- **Vocoder objective:** Train HiFi-GAN to synthesize waveforms corresponding to the target mel-spectrograms.
- **Reproducibility:** The same dataset split and evaluation procedure should be maintained across experiments for a fair comparison.

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
