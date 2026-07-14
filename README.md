# Beyond Additive Noise: Evaluating Geometric Style Transfer in Scene Text Recognition

Research project on **data augmentation for Scene Text Recognition (STR)**, carried out in collaboration with **Google** and the **EPFL LIONS laboratory** (Laboratory for Information and Inference Systems).

📄 **Full write-up: [paper.pdf](paper.pdf)**

## Overview

STR models are typically trained on synthetic data (MJSynth, SynthText) and struggle with the distortions found in real-world images — blur, noise, compression artifacts, perspective changes. This project:

1. **Benchmarks state-of-the-art STR models** (CRNN, TPS-ResNet-BiLSTM-Attn) under realistic camera-style augmentations across five real-world datasets (IIIT5k, SVT, SVTP, CUTE80, ICDAR2015).
2. **Retrains TPS-ResNet-BiLSTM-Attn (TRBA) from scratch on augmented synthetic data** (49M parameters, 41k iterations on 2× Tesla V100) to measure the robustness gained from training-time augmentation.
3. **Prototypes a DCGAN** to generate augmented street-view-style training data as a generative alternative to hand-designed augmentations.

Augmentations are deliberately kept subtle — designed to simulate phone-camera and surveillance-camera conditions without making the text illegible (15 transforms across blur, noise, camera, dropout, artistic, and geometric families).

## Key results

Sensitivity of the baseline TRBA model to augmentations at test time (accuracy drop, %):

| Dataset | Blur | Noise | Camera | Dropout | Geometric | Combined |
|---------|------|-------|--------|---------|-----------|----------|
| IIIT5k  | 5.17 | 0.17  | 0.23   | 1.70    | 7.50      | 2.73     |
| SVT     | 3.47 | 1.00  | 0.54   | 2.24    | 7.80      | 2.57     |
| SVTP    | 16.74| 0.78  | -0.16  | 4.03    | 11.63     | 5.89     |
| IC15    | 10.56| 1.41  | 0.36   | 4.06    | 9.00      | 4.68     |

After retraining TRBA on augmented data, the clean-vs-augmented accuracy gap narrows on most datasets (e.g. IIIT5k **2.73 → 1.83**, CUTE80 **4.53 → 1.74**, IC15 **4.68 → 2.05** points), showing that training-time augmentation improves robustness to real-world distortions — despite our model being trained for only 41k iterations vs. the baseline's 300k. Detailed per-dataset benchmark results are in the [results sheet](https://docs.google.com/spreadsheets/d/13Fl1kxpyHP2fxiZedBJW2poygm5o7gjkDrChHA_1edc/edit?usp=sharing) and the paper.

## Repository structure

| File | Description |
|------|-------------|
| `paper.pdf` | Full project report: methodology, training conditions, results, limitations |
| `augmentations.ipynb` | The 15 augmentation transforms and their visual effect on text images |
| `TPS_ResNet_BiLSTM_Attn_.ipynb` | TRBA model — the main SOTA model benchmarked and retrained |
| `SoA.ipynb` | State-of-the-art model evaluation across the five benchmark datasets |
| `crnn-kaggle.ipynb` | CRNN baseline (initial SOTA model), with dataset building and preprocessing |
| `DCGAN.ipynb` | DCGAN prototype for generating augmented training data |
| `AugDataset.py`, `dataset.py`, `test.py` | Dataset and evaluation utilities |
| `img_ground_truth*.csv` | Image → label mappings for IIIT5k used during initial exploration |

## Authors

Antoine Munier, Syrine Noamen — Department of Computer Science, EPFL.

Thanks to Igor Krawczuk and the LIONS laboratory for guidance and computing resources.
