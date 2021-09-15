# Ring Contrastive Learning

A PyTorch implementation of *Conditional Negative Sampling for Contrastive Learning of Visual Representations* (https://arxiv.org/abs/2010.02037).

## Abstract

Recent methods for learning unsupervised visual representations, dubbed contrastive learning, optimize the noise-contrastive estimation (NCE) bound on mutual information between two views of an image. NCE uses randomly sampled negative examples to normalize the objective. In this paper, we show that choosing difficult negatives, or those more similar to the current instance, can yield stronger representations. To do this, we introduce a family of mutual information estimators that sample negatives conditionally -- in a "ring" around each positive. We prove that these estimators lower-bound mutual information, with higher bias but lower variance than NCE. Experimentally, we find our approach, applied on top of existing models (IR, CMC, and MoCo) improves accuracy by 2-5% points in each case, measured by linear evaluation on four standard image datasets. Moreover, we find continued benefits when transferring features to a variety of new image distributions from the Meta-Dataset collection and to a variety of downstream tasks such as object detection, instance segmentation, and keypoint detection.

### Main Intuition

Negative samples in many contrastive algorithms are typically drawn i.i.d. but this may not be the best option. One way wish to vary the "difficulty" of negative samples during training in order to learn a stronger representation. While there are many ways to do this, we want to do it in a way that preserves nice properties. 

## Setup/Installation

We use Python 3, PyTorch 1.7.1, PyTorch Lightning 1.1.8, and a conda environment. Consider a variation of the commands below:

```
conda create -n ring python=3 anaconda
conda activate ring
conda install pytorch=1.7.1 torchvision -c pytorch
pip install pytorch-lightning=1.1.8
pip install tqdm dotmap
```

## Data

This repo only contains implementations for CIFAR10, CIFAR100, and STL10, all of which is in torchvision.

## Usage

For every fresh terminal instance, you should run

```
source init_env.sh
```

to add the correct paths to `sys.path` before running anything else.

The primary script is found in the `scripts/run.py` file. It is used to run pretraining and linear evaluation experiments. You must supply it a configuration file, for which many templates are in the `configs/` folder. These configuration files are not complete, you must supply a experiment base directory (`exp_base`) to point to where in your filesystem model checkpoints will go. 

Example usage:

```
python scripts/run.py config/pretrain/nce/nce.json
```

For linear evaluation, in the config file, you must provide the `exp_dir` and `checkpoint_name` (the file containing the epoch name) for the pretrained model.

## Citation
If you find this useful for your research, please cite:

```
@article{wu2020conditional,
  title={Conditional negative sampling for contrastive learning of visual representations},
  author={Wu, Mike and Mosse, Milan and Zhuang, Chengxu and Yamins, Daniel and Goodman, Noah},
  journal={arXiv preprint arXiv:2010.02037},
  year={2020}
}
```
