# SHOP
Repo containing code + dataset for Small Handheld Object Pipeline (SHOP).

- Paper [[arxiv](https://arxiv.org/abs/2203.15228)]
- [SHOP dataset](https://github.com/spider-sense/SHOP/releases/tag/0.1.0)
- [Models and test results](https://drive.google.com/drive/u/0/folders/1DbA9OkVI6kw_TNvhMKQHpfm8U9v0gzC8) (to recreate results)
- Handheld dataset creation program [<a href="https://github.com/spider-sense/handheld-classification">source code</a>] 
[[website](https://spider-sense.github.io/handheld-classification/)]

## Installation
Requirements:
- Linux (only needed if using OpenPose)

Quick installation using Anaconda/miniconda (recommended):

This will install both PyTorch and TensorFlow.

```bash
git clone https://github.com/spider-sense/SHOP.git --depth 1
cd SHOP
# Linux/macOS
conda env create -f environment.yml
# Windows
conda env create -f environment_windows.yml
```

The `--depth 1` flag is recommended to reduce git clone size.






