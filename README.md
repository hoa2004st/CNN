# DAiSEE Engagement Implementation (Start)

This repository now contains an end-to-end executable implementation for the core paper pipeline:

- DAiSEE label indexing for Engagement target
- Split integrity checks (manifest consistency + subject leakage)
- Uniform frame sampling utility for video clips
- OpenFace feature extraction module with cache key validation
- CNN frame embedding extraction module with cache key validation
- OpenFace/CNN feature fusion interface
- PCA/SVD reduction + optional SMOTE + classifier training/evaluation
- Reproducible ablation runner
- CLI entrypoints for each pipeline stage

## Project Layout

- `src/engagement_pipeline/data_index.py`: index construction and split validation
- `src/engagement_pipeline/frame_sampling.py`: deterministic temporal frame sampling
- `src/engagement_pipeline/openface.py`: OpenFace extraction and cache management
- `src/engagement_pipeline/cnn.py`: CNN embedding extraction and cache management
- `src/engagement_pipeline/fusion.py`: OpenFace/CNN alignment and fusion logic
- `src/engagement_pipeline/training.py`: clip-level pooling, reduction, SMOTE, and classifier training
- `src/engagement_pipeline/experiments.py`: default ablation suite runner
- `src/engagement_pipeline/cli.py`: command-line interface
- `artifacts/`: generated outputs (ignored by git)

## Setup

```powershell
python -m pip install -e .
```

Or use requirements file:

```powershell
python -m pip install -r requirements.txt
```

## Build DAiSEE Engagement Index

```powershell
python -m engagement_pipeline.cli build-index --daisee-root "C:\Hofa\Thesis\DAiSEE" --output-dir "artifacts\index"
```

Outputs:

- `artifacts/index/dataset_index.jsonl`: one record per clip
- `artifacts/index/index_summary.json`: diagnostics and split checks

## Sample Frames from a Clip

```powershell
python -m engagement_pipeline.cli sample-video --video-path "C:\Hofa\Thesis\DAiSEE\DataSet\Train\110001\1100011002\1100011002.avi" --num-samples 60 --output-npy "artifacts\samples\1100011002.npy"
```

## Extract and Cache OpenFace Features

Install OpenFace and ensure `FeatureExtraction` is in PATH, or pass its full executable path through `--openface-bin`.

```powershell
python -m engagement_pipeline.cli extract-openface --index-path "artifacts\index\dataset_index.jsonl" --openface-bin "FeatureExtraction" --cache-dir "artifacts\openface_cache"
```

Useful options:

- `--max-clips 20`: run a small smoke test on first 20 clips
- `--overwrite`: ignore existing cache and force re-extraction
- `--feature-flag -aus --feature-flag -pose`: replace default feature flags
- `--extra-arg -q`: pass additional OpenFace arguments

Outputs:

- `artifacts/openface_cache/<split>/<clip_stem>/features.npz`: cached feature matrix and column names
- `artifacts/openface_cache/<split>/<clip_stem>/meta.json`: cache key, source stats, and extraction config
- `artifacts/openface_cache/extraction_manifest.jsonl`: one extraction result row per clip
- `artifacts/openface_cache/extraction_summary.json`: aggregate success/failure/cache-hit statistics

## Extract and Cache CNN Features

```powershell
python -m engagement_pipeline.cli extract-cnn --index-path "artifacts\index\dataset_index.jsonl" --cache-dir "artifacts\cnn_cache" --model-name "efficientnet_b0" --device auto
```

Useful options:

- `--num-samples 60`: number of sampled frames per clip
- `--image-size 224`: resize side for each frame
- `--batch-size 16`: inference batch size
- `--no-pretrained`: disable ImageNet weights
- `--max-clips 20`: run a short smoke test

Outputs:

- `artifacts/cnn_cache/<split>/<clip_stem>/features.npz`: cached per-frame embedding matrix
- `artifacts/cnn_cache/<split>/<clip_stem>/meta.json`: cache key, source stats, and extraction config
- `artifacts/cnn_cache/extraction_manifest.jsonl`: one extraction result row per clip
- `artifacts/cnn_cache/extraction_summary.json`: aggregate success/failure/cache-hit statistics

## Fuse OpenFace and CNN Features

```powershell
python -m engagement_pipeline.cli fuse-features --index-path "artifacts\index\dataset_index.jsonl" --openface-cache-dir "artifacts\openface_cache" --cnn-cache-dir "artifacts\cnn_cache" --output-dir "artifacts\fused_cache" --alignment-mode truncate --fusion-method concat
```

Useful options:

- `--alignment-mode truncate|pad_repeat_last|interpolate_max`
- `--fusion-method concat|add`
- `--max-clips 20`: run a short smoke test

Outputs:

- `artifacts/fused_cache/<split>/<clip_stem>/features.npz`: fused per-frame feature matrix
- `artifacts/fused_cache/<split>/<clip_stem>/meta.json`: source paths, source shapes, and fusion config
- `artifacts/fused_cache/fusion_manifest.jsonl`: one fusion result row per clip
- `artifacts/fused_cache/fusion_summary.json`: aggregate success/failure/cache-hit statistics

## Train Classifier (PCA/SVD + Optional SMOTE)

```powershell
python -m engagement_pipeline.cli train-classifier --index-path "artifacts\index\dataset_index.jsonl" --feature-cache-dir "artifacts\fused_cache" --output-dir "artifacts\training" --pooling-mode mean --reduction-method pca --n-components 256 --classifier-name logistic_regression --class-weight balanced --enable-smote
```

Useful options:

- `--feature-cache-dir artifacts/openface_cache`: train using OpenFace-only features
- `--feature-cache-dir artifacts/cnn_cache`: train using CNN-only features
- `--reduction-method none|pca|svd`
- `--classifier-name logistic_regression|linear_svm|random_forest|mlp`
- `--allow-missing-features`: continue if some cached feature files are missing

Outputs:

- `artifacts/training/model_bundle.joblib`: persisted scaler/reducer/classifier bundle
- `artifacts/training/feature_manifest.jsonl`: feature-loading result for each clip
- `artifacts/training/train_summary.json`: split metrics, confusion matrices, transform settings, and SMOTE report

## Run Default Ablation Suite

```powershell
python -m engagement_pipeline.cli run-ablations --index-path "artifacts\index\dataset_index.jsonl" --openface-cache-dir "artifacts\openface_cache" --cnn-cache-dir "artifacts\cnn_cache" --fused-cache-dir "artifacts\fused_cache" --output-dir "artifacts\experiments" --pooling-mode mean --classifier-name logistic_regression --n-components 256
```

Default ablation runs include:

- OpenFace-only baseline
- CNN-only baseline
- Fused without reduction
- Fused + PCA
- Fused + SVD
- Fused + PCA + SMOTE

Outputs:

- `artifacts/experiments/ablation_results.csv`: compact table for quick comparison
- `artifacts/experiments/ablation_summary.json`: full run metadata and status
- `artifacts/experiments/<run_name>/`: per-run training artifacts

## Rented Server Next Steps

1. SSH into the rented GPU server and create a persistent session (`tmux new -s daisee`).
2. Clone repository and install dependencies.
3. Download or copy DAiSEE to server and confirm `DataSet/` and `Labels/` exist.
4. Build index with `build-index` and verify split counts in `artifacts/index/index_summary.json`.
5. Run `extract-openface` (requires OpenFace `FeatureExtraction` binary available in PATH).
6. Run `extract-cnn` and check `artifacts/cnn_cache/extraction_summary.json`.
7. Run `fuse-features` and confirm non-zero success count in `artifacts/fused_cache/fusion_summary.json`.
8. Run `train-classifier` for your chosen setup and inspect validation/test macro-F1 in `artifacts/training/train_summary.json`.
9. Run `run-ablations` and review `artifacts/experiments/ablation_results.csv` to pick the best configuration.
10. Download only the needed artifacts (`training/`, `experiments/`, selected cache summaries) back to local machine.
