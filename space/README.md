---
title: MelodAI
emoji: 🎵
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: What a neural network can and cannot hear in a Hindi film song.
---

# MelodAI

Energy classification over YAMNet embeddings, with the evidence shown rather
than summarised: the per window energy curve, the AudioSet classes the network
assigns, nearest songs in the corpus, placement on the corpus map, and the
three channel spectrogram the convolutional model was trained on.

Serving runs without TensorFlow. YAMNet is a vendored TFLite build and the
energy head is four matrix multiplies in NumPy, checked against the original
Keras model to 1e-7.

Source and method: https://github.com/OmegaKavya/MelodAI
