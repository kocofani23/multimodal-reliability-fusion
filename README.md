# Robust Image-Text Fusion with Pretrained CLIP Features

## Overview

This project studies multi-label classification from paired images and captions. It uses frozen image and text encoders from pretrained CLIP to create feature vectors, then trains lightweight classifiers to combine those vectors.

Multimodal systems can fail when one input is absent or unreliable. A caption may be missing, an image representation may be corrupted, or one modality may be much less informative than the other. The project compares late fusion, late fusion trained with modality and feature perturbations, and a learned reliability-aware gate under clean, missing-modality, and image-feature-corruption conditions.

## Research Question and Objectives

**Research question:** Can a learned reliability-aware fusion mechanism improve robustness to missing and corrupted image or text features compared with standard late fusion and ModDrop-based approaches, while maintaining good classification performance on clean data?

The experiments aim to:

1. Establish clean-input and robustness baselines for late fusion.
2. Measure the effect of training-time modality masking and feature-noise augmentation.
3. Evaluate whether a learned gate can adapt to missing or corrupted inputs.
4. Compare the learned gate with an oracle gate supplied with condition-level reliability information.
5. Report performance across three random seeds.

## Methodology

1. **Prepare paired COCO samples.** COCO image IDs associate each image with its captions and object-instance annotations. The loader keeps image IDs that have captions and an image file on disk.
2. **Construct targets.** For each image, the 80 COCO detection categories are sorted by category ID and represented as a multi-hot vector: a category is 1 if at least one instance of it appears in the image.
3. **Extract CLIP representations.** The `openai/clip-vit-base-patch32` image and text encoders are loaded in evaluation/inference mode. Captions are tokenized with the matching CLIP tokenizer; the CLIP image processor handles image preprocessing. The encoders are not fine-tuned.
4. **Pool captions.** Every caption for an image is encoded, each caption embedding is L2-normalized, and the embeddings are averaged per image and normalized again.
5. **Train fusion classifiers.** The evaluator splits the saved features, trains the configured fusion variants with binary cross-entropy, selects a global classification threshold using clean validation macro-F1, and evaluates the test split under each condition.
6. **Measure performance.** The evaluator reports macro-F1, micro-F1, and mean average precision (mAP). The threshold tuned on clean validation data is reused for every test condition.

Feature corruption is applied to saved embeddings rather than raw images or caption text. In the implemented evaluation sweep, the corrupted modality is the image feature; text-feature corruption is supported by the inference function but is not included in the named saved test conditions.

## Model Architectures

All models classify paired 512-dimensional feature vectors and output 80 class logits. An explicit mask means “unavailable”; the model zeros that feature before fusion. The models do not perform random modality dropout internally.

### Late fusion (`A_latefusion_noaug`)

**Purpose:** Baseline joint classification from both modalities.

The model concatenates the image and text features (1,024 dimensions total) and feeds them to a two-layer MLP with a 256-unit hidden layer. Missing modalities are zeroed before concatenation. This variant is trained without missing-modality or feature-noise augmentation.

### Late fusion with ModDrop-style augmentation (`B_latefusion_moddrop`)

**Purpose:** Test whether training with simulated modality failures and feature noise improves robustness without changing the fusion architecture.

It uses the same concatenate-and-classify model as the baseline. During training, samples may have one modality masked and image or text feature noise added. The masking and noise are input augmentations, not a separate dropout layer in the model. At evaluation, it handles missing inputs through the same explicit masks as the baseline.

### Learned gated concatenation (`C_gated_concat`)

**Purpose:** Learn a per-sample relative weight for each modality before classification.

A small gate network receives the masked image and text features plus two availability indicators. A two-way softmax produces image and text weights. Each weight scales every feature dimension of its modality; the two weighted vectors are then concatenated and classified. Missing modalities are zeroed and their availability is exposed to the gate. Present-but-corrupted features are not explicitly identified as corrupted; the gate only sees their altered values.

This model is trained with the same masking/noise augmentation as the ModDrop-style variant, and includes auxiliary per-modality classification losses during training.

### Oracle gated concatenation (`D_oracle_gated_concat`)

**Purpose:** Provide an informed-weight comparison for the gated architecture.

It uses the same gated-concatenation classifier, but evaluation supplies condition-derived weights instead of using the learned gate. A missing modality receives reliability zero; otherwise reliability is `exp(-3 × sigma)`, where `sigma` is the configured feature-noise standard deviation. The two reliabilities are normalized to sum to one. This variant therefore assumes the evaluation condition and noise level are known; it is not a learned corruption detector.


## Project Structure
.
├── comparisons.txt
├── results.txt
├── data/
│   ├── processed/
│   │   └── extracted_features.pt
│   └── raw/
│       ├── annotations/
│       │   ├── captions_train2017.json
│       │   ├── captions_val2017.json
│       │   ├── instances_train2017.json
│       │   ├── instances_val2017.json
│       │   ├── person_keypoints_train2017.json
│       │   └── person_keypoints_val2017.json
│       └── val2017/                 # 5,000 COCO validation images
├── experiments/
│   ├── checkpoints/                 # 12 checkpoints: 4 variants × 3 seeds
│   └── results/
│       ├── aggregated_results.json
│       ├── config.json
│       ├── per_seed_results.json
│       └── training_losses.json
└── src/
    ├── data_loader.py
    ├── evaluate.py
    ├── features.py
    ├── models.py
    └── show_gate.py

- `src/data_loader.py` joins image files, caption annotations, and instance annotations into COCO multimodal samples.
- `src/features.py` extracts and saves CLIP features, labels, image IDs, and extraction provenance.
- `src/models.py` defines late-fusion, gated-concatenation, and gated-sum classifiers.
- `src/evaluate.py` contains the training, validation-threshold selection, evaluation, and result aggregation pipeline.
- `src/show_gate.py` prints the saved gate-weight summaries for the learned and oracle gated models.
- `data/processed/extracted_features.pt` contains the cached feature dataset used by the evaluator.
- `experiments/results/config.json` records the effective settings of the saved run. There is no `experiments/configs/` directory; the evaluator defaults and named conditions are defined in `src/evaluate.py`.
- `comparisons.txt` and `results.txt` contain saved comparison notes and run output. The JSON files under `experiments/results/` are the structured result artifacts.

## Dataset

The project uses the [COCO dataset](https://cocodataset.org/) and its [download page](https://cocodataset.org/#download). COCO provides images, captions, and object-instance annotations. The project joins these annotation types using the shared `image_id`.

The loader uses every caption associated with an image, not just one caption. Instance annotations are converted to 80-dimensional multi-hot category-presence targets; object counts are not used. Images are converted to RGB, then preprocessed by the CLIP image processor during feature extraction. The loader filters out images without captions and files missing from the image directory. It does not otherwise filter samples by target labels.

The checked-in data contains the 5,000 `val2017` images and their caption and instance annotations. The saved feature artifact contains 5,000 corresponding rows. Although train and validation COCO annotation JSON files are present, the feature-extraction defaults point to `val2017`, `captions_val2017.json`, and `instances_val2017.json`.

For each seed, the evaluator randomly permutes these 5,000 feature rows and creates non-stratified 70%/15%/15% training, validation, and test splits: 3,500 / 750 / 750 samples. These are splits of COCO `val2017`, not the official COCO training/validation/test sets. The seed-specific splits are generated independently.

## Requirements and Installation

There is no `requirements.txt`, `environment.yml`, or `pyproject.toml` in the repository. The source imports PyTorch, NumPy, Pillow, Transformers, tqdm, and scikit-learn. A checked-in Python bytecode cache is tagged for CPython 3.12, but the project does not declare a supported Python version. The saved feature provenance records PyTorch `2.10.0+cpu` and Transformers `5.20.1` for the existing extraction; other dependency versions are not recorded.


## Running the Project

Run commands from the repository root.

### 1. Prepare the dataset

Download COCO `val2017` images and the matching 2017 caption and instance annotations from the [COCO download page](https://cocodataset.org/#download). Place them at:

data/raw/val2017/
data/raw/annotations/captions_val2017.json
data/raw/annotations/instances_val2017.json

These are the paths used by default in `src/features.py`. The loader requires an image file and at least one caption for an image to retain it. Dataset downloads and placement are manual; the repository contains no preparation script.

### 2. Extract CLIP features

python src/features.py

This uses batch size 64 by default and writes `data/processed/extracted_features.pt`. Optional command-line arguments are `--batch_size` and `--limit`, for example `python src/features.py --limit 100` for a small feature-extraction check. The dataset paths and model name are function defaults, not command-line options.

### 3. Train the fusion models and run evaluation

python src/evaluate.py

This runs the four configured variants for seeds `42,43,44`, trains for 20 epochs, and writes checkpoints and result files under `experiments/checkpoints/` and `experiments/results/`. Training and test evaluation are combined in this script; there is no separate training entry point.

Useful evaluator options are:

python src/evaluate.py --seeds 42 --epochs 20 --batch_size 64


### 4. Inspect the results

- `experiments/results/aggregated_results.json` contains mean and standard deviation across the requested seeds.
- `experiments/results/per_seed_results.json` contains individual seed metrics and gate summaries.
- `experiments/results/training_losses.json` contains epoch loss histories.
- `experiments/results/config.json` records the effective settings, conditions, model names, seeds, and selected device.
- `python src/show_gate.py` prints gate-weight summaries from `per_seed_results.json`.

No figure-generation script or saved figures are present in the repository.

## Experimental Setup

The saved configuration uses:

| Setting | Value |
|---|---:|
| Seeds | 42, 43, 44 |
| Epochs | 20 |
| Batch size | 64 |
| Optimizer / learning rate | Adam / 0.001 |
| Loss | Binary cross-entropy with logits |
| Train / validation / test fractions | 70% / 15% / 15% |
| Augmentation masking probability | 0.10 per modality |
| Image feature corruption probability during training | 0.15 |
| Text feature corruption probability during training | 0.075 |
| Maximum sampled image / text noise scale during training | 0.15 / 0.075 |
| Auxiliary gated-model loss weight | 0.3 |
| Validation threshold candidates | 0.10 to 0.90 in increments of 0.05 |

During training augmentation, each sample is assigned to image-missing, text-missing, or neither with probabilities 0.10, 0.10, and 0.80. Feature corruption is sampled independently of masking. For selected samples, Gaussian noise standard deviation is sampled uniformly from `[0, 0.15)` for image features and `[0, 0.075)` for text features.

At test time, the saved evaluation conditions are:

| Condition | Image masked | Text masked | Image noise standard deviation | Text noise standard deviation |
|---|---:|---:|---:|---:|
| Clean | No | No | 0 | 0 |
| Image missing | Yes | No | 0 | 0 |
| Text missing | No | Yes | 0 | 0 |
| Image corruption | No | No | 0.05, 0.10, 0.20, 0.30, or 0.50 | 0 |

Noise is additive, zero-mean Gaussian noise applied to feature dimensions. Corrupted features are not renormalized after perturbation. No saved test condition corrupts text features or masks both modalities.

The reported classification metrics are macro-F1, micro-F1, and mAP. A single global threshold is selected by maximizing macro-F1 on clean validation predictions, then kept fixed for all test conditions. The mAP calculation averages average precision over classes with at least one positive test target; the saved per-seed files indicate 79 classes were included and one class was skipped.


## Results

The following values are the saved mean macro-F1 ± standard deviation across three seeds. They come from `experiments/results/aggregated_results.json`; they are not newly run measurements.

| Model | Clean | Image missing | Text missing | Image noise σ=0.20 | Image noise σ=0.50 |
|---|---:|---:|---:|---:|---:|
| Late fusion, no augmentation | 0.6429 ± 0.0102 | 0.6104 ± 0.0118 | 0.4826 ± 0.0060 | 0.3759 ± 0.0083 | 0.1516 ± 0.0062 |
| Late fusion + ModDrop-style augmentation | **0.6593 ± 0.0136** | **0.6268 ± 0.0231** | 0.3986 ± 0.0432 | 0.4786 ± 0.0100 | 0.2079 ± 0.0045 |
| Learned gated concatenation | 0.6282 ± 0.0081 | 0.6176 ± 0.0001 | 0.5360 ± 0.0333 | 0.4771 ± 0.0113 | 0.2830 ± 0.0370 |
| Oracle gated concatenation | 0.6301 ± 0.0100 | 0.6196 ± 0.0045 | **0.5432 ± 0.0297** | **0.5329 ± 0.0029** | **0.5089 ± 0.0093** |

**Measured observations:** ModDrop-style late fusion has the strongest clean macro-F1 and image-missing macro-F1 among the four tested variants. The learned and oracle gated models score higher than both late-fusion variants when text is missing. Under severe image-feature corruption, the learned gate outperforms both late-fusion variants, while the oracle gate is substantially stronger.

**Interpretation and limitations:** These results show a clean-performance/robustness trade-off rather than a universal win for learned gating: the learned gated model's clean macro-F1 is below both late-fusion variants, and its performance falls as image noise increases. The oracle's advantage under corruption depends on being given the known noise level; it should not be interpreted as evidence that the learned gate detects corruption. These measurements are from random splits of COCO `val2017` features and do not establish performance on the official COCO training or test sets.

The aggregated result file also reports micro-F1 and mAP for every model-condition pair. In the clean condition, the saved mAP means are 0.7081 (no-augmentation late fusion), 0.7315 (ModDrop-style late fusion), 0.6972 (learned gate), and 0.6968 (oracle gate). Consult the JSON output for all conditions and metric standard deviations.