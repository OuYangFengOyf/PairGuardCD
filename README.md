<div align="center">
<h1>PairGuard-CD</h1>
<h3>Reliable Offline Supervision and Relation-Guided Lightweight Change Detection with Limited Labels</h3>
<img src="https://img.shields.io/badge/Python-3.9%2B-blue">
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c">
<img src="https://img.shields.io/badge/Task-Remote%20Sensing%20Change%20Detection-green">
<img src="https://img.shields.io/badge/Inference-Student%20Only-orange">
</div>
:page_with_curl: Overview
PairGuard-CD: Reliable Offline Supervision and
Relation-Guided Lightweight Change Detection with Limited
Labels

:golf: Getting Started
1. Environment
```bash
conda env create -f environment.yaml
conda activate pairguard
pip install -e .
```
or
```bash
pip install -r requirements.txt
pip install -e .
```
2. Foundation Models
Install SAM2 and prepare SAM2.1-Hiera-Large:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```
DINOv2-S/14 is loaded through:
```python
DINOv2FeatureExtractor.from_torch_hub("dinov2_vits14")
```
3. Dataset Manifest
Each line uses:
```text
I1,I2,GT,CACHE,GROUP_ID
```
Example:
```text
data/A/000001.png,data/B/000001.png,data/label/000001.png,,scene_000001
```
Create a limited-label split:
```bash
python scripts/create_label_splits.py \
  --manifest data/train_all.txt \
  --ratio 0.20 \
  --seed 2026 \
  --labeled-out data/train_labeled.txt \
  --unlabeled-out data/train_unlabeled.txt
```
:construction: Offline Teacher
Generate 5-fold OOF seeds:
```bash
python scripts/train_oof_seeds.py \
  --labeled-manifest data/train_labeled.txt \
  --unlabeled-manifest data/train_unlabeled.txt \
  --out-dir outputs/oof_seeds \
  --folds 5
```
Build teacher manifests and NARP-US records:
```bash
python scripts/build_teacher_manifests.py \
  --labeled-manifest data/train_labeled.txt \
  --unlabeled-manifest data/train_unlabeled.txt \
  --seed-dir outputs/oof_seeds \
  --narp-out data/narp_labeled.txt \
  --teacher-out data/unlabeled_teacher.txt

python scripts/collect_narp_records.py \
  --manifest data/narp_labeled.txt \
  --out outputs/narp_records.npz \
  --sam2-config sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda
```
Train NARP-US and build the supervision cache:
```bash
python scripts/train_narp_selector.py \
  --records outputs/narp_records.npz \
  --out outputs/narp_selector.pt \
  --folds 5 \
  --device cuda

python scripts/build_cache.py \
  --manifest data/unlabeled_teacher.txt \
  --out-dir outputs/teacher_cache \
  --output-manifest data/train_unlabeled_cached.txt \
  --sam2-config sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --selector-checkpoint outputs/narp_selector.pt \
  --device cuda
```
:rocket: Training
Stage S0:
```bash
python scripts/train_pairguard.py \
  --stage s0 \
  --labeled-manifest data/train_labeled.txt \
  --val-manifest data/val.txt \
  --epochs 40 \
  --save checkpoints/s0.pt
```
Stage S1:
```bash
python scripts/train_pairguard.py \
  --stage s1 \
  --labeled-manifest data/train_labeled.txt \
  --unlabeled-manifest data/train_unlabeled_cached.txt \
  --val-manifest data/val.txt \
  --resume checkpoints/s0.pt \
  --epochs 100 \
  --save checkpoints/s1.pt
```
Stage S2:
```bash
python scripts/train_pairguard.py \
  --stage s2 \
  --labeled-manifest data/train_labeled.txt \
  --val-manifest data/val.txt \
  --resume checkpoints/s1.pt \
  --epochs 30 \
  --save checkpoints/s2.pt
```
:mag: Evaluation
```bash
python scripts/evaluate.py \
  --manifest data/test.txt \
  --checkpoint checkpoints/s2.pt \
  --device cuda
```
Full-resolution sliding-window evaluation:
```bash
python scripts/evaluate_sliding.py \
  --manifest data/test_fullres.txt \
  --checkpoint checkpoints/s2.pt \
  --crop-size 256 \
  --stride 128 \
  --device cuda
```
:zap: Efficiency
```bash
python scripts/benchmark_inference.py \
  --checkpoint checkpoints/s2.pt \
  --size 256 \
  --warmup 100 \
  --iters 1000 \
  --device cuda
```
:open_file_folder: Repository Structure
```text
PairGuard-CD/
├── pairguard/
│   ├── models/
│   ├── offline_teacher/
│   └── training/
├── scripts/
├── tests/
├── configs/
├── environment.yaml
├── pyproject.toml
└── requirements.txt
```
:sunrise: Acknowledgements
This project uses SAM2, DINOv2, and torchvision.
