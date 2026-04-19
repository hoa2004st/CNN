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
- One-shot paper runner for full-dataset or sharded execution
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
- `scripts/run_paper_pipeline.py`: orchestrate the full paper pipeline into one output root
- `outputs/`: durable generated outputs (not ignored by git)
- `artifacts/`: scratch/generated outputs from older runs or ad hoc local experiments (ignored by git)

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
python -m engagement_pipeline.cli build-index --daisee-root "C:\Hofa\Thesis\DAiSEE" --output-dir "outputs\index"
```

Outputs:

- `outputs/index/dataset_index.jsonl`: one record per clip
- `outputs/index/index_summary.json`: diagnostics and split checks

## Sample Frames from a Clip

```powershell
python -m engagement_pipeline.cli sample-video --video-path "C:\Hofa\Thesis\DAiSEE\DataSet\Train\110001\1100011002\1100011002.avi" --num-samples 60 --output-npy "outputs\samples\1100011002.npy"
```

## Extract and Cache OpenFace Features

Install OpenFace and ensure `FeatureExtraction` is in PATH, or pass its full executable path through `--openface-bin`.

```powershell
python -m engagement_pipeline.cli extract-openface --index-path "outputs\index\dataset_index.jsonl" --openface-bin "FeatureExtraction" --cache-dir "outputs\features\openface_cache"
```

Useful options:

- `--max-clips 20`: run a small smoke test on first 20 clips
- `--split train --max-clips-per-split 200`: shard long extraction runs by split or cap clips per split
- `--overwrite`: ignore existing cache and force re-extraction
- `--feature-flag -aus --feature-flag -pose`: replace default feature flags
- `--extra-arg -q`: pass additional OpenFace arguments

Outputs:

- `outputs/features/openface_cache/<split>/<clip_stem>/features.npz`: cached feature matrix and column names
- `outputs/features/openface_cache/<split>/<clip_stem>/meta.json`: cache key, source stats, and extraction config
- `outputs/features/openface_cache/extraction_manifest.jsonl`: one extraction result row per clip
- `outputs/features/openface_cache/extraction_summary.json`: aggregate success/failure/cache-hit statistics

## Extract and Cache CNN Features

```powershell
python -m engagement_pipeline.cli extract-cnn --index-path "outputs\index\dataset_index.jsonl" --cache-dir "outputs\features\cnn_cache" --model-name "efficientnet_b0" --device auto
```

Useful options:

- `--num-samples 60`: number of sampled frames per clip
- `--image-size 224`: resize side for each frame
- `--batch-size 16`: inference batch size
- `--no-pretrained`: disable ImageNet weights
- `--split validation --max-clips-per-split 100`: run a bounded extraction job on a subset

Outputs:

- `outputs/features/cnn_cache/<split>/<clip_stem>/features.npz`: cached per-frame embedding matrix
- `outputs/features/cnn_cache/<split>/<clip_stem>/meta.json`: cache key, source stats, and extraction config
- `outputs/features/cnn_cache/extraction_manifest.jsonl`: one extraction result row per clip
- `outputs/features/cnn_cache/extraction_summary.json`: aggregate success/failure/cache-hit statistics

## Fuse OpenFace and CNN Features

```powershell
python -m engagement_pipeline.cli fuse-features --index-path "outputs\index\dataset_index.jsonl" --openface-cache-dir "outputs\features\openface_cache" --cnn-cache-dir "outputs\features\cnn_cache" --output-dir "outputs\features\fused_cache" --alignment-mode truncate --fusion-method concat
```

Useful options:

- `--alignment-mode truncate|pad_repeat_last|interpolate_max`
- `--fusion-method concat|add`
- `--split train --max-clips-per-split 500`: fuse only a selected shard

Outputs:

- `outputs/features/fused_cache/<split>/<clip_stem>/features.npz`: fused per-frame feature matrix
- `outputs/features/fused_cache/<split>/<clip_stem>/meta.json`: source paths, source shapes, and fusion config
- `outputs/features/fused_cache/fusion_manifest.jsonl`: one fusion result row per clip
- `outputs/features/fused_cache/fusion_summary.json`: aggregate success/failure/cache-hit statistics

## Train Classifier (PCA/SVD + Optional SMOTE)

```powershell
python -m engagement_pipeline.cli train-classifier --index-path "outputs\index\dataset_index.jsonl" --feature-cache-dir "outputs\features\fused_cache" --output-dir "outputs\training" --pooling-mode mean --reduction-method pca --n-components 256 --classifier-name logistic_regression --class-weight balanced --enable-smote
```

Useful options:

- `--feature-cache-dir outputs/features/openface_cache`: train using OpenFace-only features
- `--feature-cache-dir outputs/features/cnn_cache`: train using CNN-only features
- `--reduction-method none|pca|svd`
- `--classifier-name logistic_regression|linear_svm|random_forest|mlp`
- `--allow-missing-features`: continue if some cached feature files are missing
- `--max-clips-per-split 250`: fast sanity-check run with train/validation/test all represented

Outputs:

- `outputs/training/model_bundle.joblib`: persisted scaler/reducer/classifier bundle
- `outputs/training/feature_manifest.jsonl`: feature-loading result for each clip
- `outputs/training/train_summary.json`: split metrics, confusion matrices, label distributions, transform settings, and SMOTE report

## Run Default Ablation Suite

```powershell
python -m engagement_pipeline.cli run-ablations --index-path "outputs\index\dataset_index.jsonl" --openface-cache-dir "outputs\features\openface_cache" --cnn-cache-dir "outputs\features\cnn_cache" --fused-cache-dir "outputs\features\fused_cache" --output-dir "outputs\experiments" --pooling-mode mean --classifier-name logistic_regression --n-components 256
```

Default ablation runs include:

- OpenFace-only baseline
- CNN-only baseline
- Fused without reduction
- Fused + PCA
- Fused + SVD
- Fused + PCA + SMOTE

Outputs:

- `outputs/experiments/ablation_results.csv`: compact table for quick comparison
- `outputs/experiments/ablation_summary.json`: full run metadata, status, and best-run selection
- `outputs/experiments/<run_name>/`: per-run training artifacts

## Run The Full Paper Pipeline

This wraps indexing, OpenFace extraction, CNN extraction, fusion, fused-model training, and the default ablation suite into one reproducible output root:

```powershell
python scripts/run_paper_pipeline.py --daisee-root "data\DAiSEE" --output-root "outputs\paper_run" --openface-bin "FeatureExtraction" --cnn-device auto --enable-smote
```

Useful options:

- `--skip-openface|--skip-cnn|--skip-fusion|--skip-training|--skip-ablations`: resume from existing caches
- `--split train --max-clips-per-split 1000`: shard long jobs for rented servers
- `--feature-root "outputs\shared_features"`: place all extracted OpenFace/CNN/fused features in one explicit folder
- `--overwrite`: force regeneration instead of reusing cache

Primary outputs:

- `outputs/paper_run/run_summary.json`: top-level record of the run
- `outputs/paper_run/index/`: generated dataset index and diagnostics
- `outputs/paper_run/features/openface_cache/`, `cnn_cache/`, `fused_cache/`: feature caches by default
- `outputs/paper_run/training/`: main fused training run
- `outputs/paper_run/experiments/`: ablation suite

## Rented Server Next Steps

1. SSH into the rented GPU server and create a persistent session (`tmux new -s daisee`).
2. Clone repository and install dependencies.
3. Download or copy DAiSEE to server and confirm `DataSet/` and `Labels/` exist.
4. Build index with `build-index` and verify split counts in `outputs/index/index_summary.json`.
5. Prefer `python scripts/run_paper_pipeline.py --daisee-root "data/DAiSEE" --output-root "outputs/paper_run" --openface-bin "FeatureExtraction"` for a full run.
6. If the full run is too long for one session, shard it with `--split ...` and `--max-clips-per-split ...`, then resume with `--skip-*` flags.
7. Check `outputs/paper_run/run_summary.json` after each stage to verify non-zero success counts.
8. Inspect `outputs/paper_run/training/train_summary.json` for validation/test macro-F1 and class distribution.
9. Review `outputs/paper_run/experiments/ablation_results.csv` and `ablation_summary.json` to choose the best configuration.
10. Download only the needed artifacts (`training/`, `experiments/`, selected cache summaries) back to local machine.
