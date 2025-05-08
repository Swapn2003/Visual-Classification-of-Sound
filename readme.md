# Multimodal Analysis and Classification of Environmental Audio Using Spectrogram Representations

## Project Overview

This project explores how environmental audio can be classified by converting audio signals into spectrogram representations and leveraging Convolutional Neural Networks (CNNs) — originally designed for image classification. This approach connects the audio and visual modalities, showcasing a practical case of **multimodal learning**.

## Motivation

Environmental sound classification has a wide range of applications — from **smart surveillance systems** and **urban sound tagging** to **wildlife monitoring** and **disaster detection**. While most of our daily ML exposure focuses on images or text, **environmental audio** offers a rich, often underexplored source of information.

I chose this topic to explore how **audio signals can be treated like images** using spectrograms and how **deep learning models from computer vision** can be adapted for audio classification. It also allowed me to step into **multimodal learning**, where different types of data (like audio, video, and text) are fused or transformed across domains.

## Multimodal Learning Perspective

Multimodal learning involves **integrating and processing multiple data modalities**. In this project, I used **spectrograms** to convert **1D audio signals into 2D image-like representations**, bridging audio and vision domains.

### Recent Milestones:

* **VGGish (Google)**: Converts audio into embeddings similar to CNN-based image features.


This project builds on that trajectory, demonstrating how **vision models (CNNs)** can be repurposed for **audio classification** using spectrograms as input.

## Objectives

- Convert environmental audio into spectrogram images
- Train a CNN-based classifier (custom or pretrained like ResNet) on spectrograms
- Achieve high classification accuracy on the UrbanSound8K dataset
- Reflect on the multimodal learning insights gained from this process


## Dataset

- **Dataset Used:** UrbanSound8K
- 10 environmental sound classes:
  - air_conditioner
  - car_horn
  - children_playing
  - dog_bark
  - drilling
  - engine_idling
  - gun_shot
  - jackhammer
  - siren
  - street_music
- Preprocessing done via **Mel Spectrogram** transformation using `librosa` and `torchaudio`

## Methodology & Architecture

### Data Preprocessing

* Converted `.wav` files into **Mel Spectrograms** using `librosa` and `torchaudio`
* Normalized and resized spectrograms to a consistent shape suitable for CNN input

```python
import torchaudio.transforms as T

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=128,
    n_fft=1024,
    hop_length=512
)
spec = mel_spectrogram(waveform)
```

### Model

- **Architecture:** Custom CNN 
- **Input:** Spectrograms (1 channel images)
- **Accuracy Achieved:** ~98% on test set

## Results

- High classification accuracy across all environmental sound classes
- Visualized spectrograms and predictions
- Robust performance across noisy and clean audio samples
- Confusion matrix showing strong class-wise performance


## Reflections

### What Surprised Me?

* Spectrograms retained **enough discriminative patterns** to allow vision-based CNNs to perform so well.
* The effectiveness of audio-to-image conversion opened ideas for **cross-modal transfer learning**.

### Scope for Improvement and Future Scope

* Incorporate **temporal context** using RNNs or Transformers after CNN feature extraction.
* Use **multi-modal fusion** (e.g., pairing audio with video or text).


## References

* [1] [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/)


## Author

**Swapn Agarwal**

Feel free to connect or open an issue if you have questions or suggestions!