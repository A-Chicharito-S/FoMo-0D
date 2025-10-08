# FoMo-0D
Official implementation of [FoMo-0D](https://arxiv.org/abs/2409.05672), a foundation model for zero-shot tabular outlier detection.

## Setup Instructions
- please first follow the instructions in [DTE](https://github.com/vicliv/DTE/tree/main) (https://arxiv.org/abs/2305.18593) and [PFN](https://github.com/automl/PFNs) (https://arxiv.org/abs/2112.10510)
- after setting up the necessary packages for DTE and PFN, in case the CPU-version of PyTorch is installed, we can install a compatible [PyTorch](https://pytorch.org/get-started/locally/) version (e.g., GPU) to our own machine
- to pre-train and evaluate FoMo-0D, we need the installation of [pytorch-lightning](https://pypi.org/project/pytorch-lightning/) for training, [hydra](https://hydra.cc/docs/intro/) and [omegaconf](https://pypi.org/project/omegaconf/) for hyperparameter management, and [wandb](https://pypi.org/project/wandb/) for monitoring training dynamics.

## Structure of the Code
**Note** on naming: we support `xxx_torch.py` as the GPU accelerated version (faster but consumes more GPU memory) of the original `xxx.py` (which uses CPU to generate data, slower when generating data), and since `pytorch-lightning` supports multi-GPU training, we use `xxx_parallel.py` to indicate that.
- \configuration: contains `yaml` files for `train`, `test`, and `prior` data generation, which can be changed in the command line in a dictionary-alike grammar (see [hydra](https://hydra.cc/docs/intro/))
- \data_prior: includes code for synthetic data generation
  - `GMM.py` and `GMM_torch.py`: contain GMM class that draws inlier and outliers
  - `parallel_generator.py` and `parallel_generator_torch.py`: contain `PriorTrainDataGenerator` class that draws synthetic data for each epoch, and `def get_batch_for_NdMclusterGaussian` that serves as the `collate_fn` for PyTorch DataLoader
- \DTE: adapted from [DTE](https://github.com/vicliv/DTE/tree/main), include code and results for baselines
- \pfns: adapted from [PFN](https://github.com/automl/PFNs), include code for the backbone transformer
  - `transformer.py`: contains model definition. In `def init_weights`, we follow the original implementation of PFN and initialize the weights for output matrices as zero for the MHAs in the RouterAttention (via `def init_router`)
  - `layer.py`: contains the definition of each layer and `RouterAttention`. 
  - `parallel_dataset.py` and `parallel_dataset_torch.py`: contain definition of `EpochDataset` to draw different datasets across epochs
  - `train_parallel.py` and `train_parallel_torch.py`: contain the pytorch-lightning module for training and evaluation
- `pretrain_parallel.py` and `pretrain_parallel_torch.py`: contain the code to train and save FoMo-0D
- `zero_shot_on_adbench.py`: contains the code to evaluate FoMo-0D zero-shot on [ADBench](https://arxiv.org/abs/2206.09426)
- `sort_zero_shot_result.py`: contains the code to sort out FoMo-0D zero-shot results
- `wilcoxon_on_adbench.py`: contains the code the do p-value test


## Pretrain FoMo-0D

To pretrain our model, the parameters are:

`prior.gmm.max_feature_dim` and `prior.gmm.max_model_dim`: input feature of the model

`train.reuse_data_every_n`: reuse at epoch-level (how many epochs to reuse)

`train.apply_linear_transform`: whether to apply linear transformation (LT) on the synthetic data

`train.gen_one_train_one`: whether to generate data on-the-fly, if `True`, ignores `train.reuse_data_every_n`

(below we pretrain FoMo-0D with `input_feature=100`, `generate-on-the-fly`, and `no-LT`)
- **generate validation data**: 
  - `CUDA_VISIBLE_DEVICES=0 python data_prior/parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500` 
  - or `CUDA_VISIBLE_DEVICES=0 python data_prior/parallel_generator_torch.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500`, 

  and the generated data wil be under `prior.gmm.data_dir` (default is `'syn_data/INLA_GMM'`)
- **pretrain**: 
  - `CUDA_VISIBLE_DEVICES=0 python pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500` 
  - or `CUDA_VISIBLE_DEVICES=0 python pretrain_parallel_torch.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500`, 
  
  and a saving directory example is `'ckpt/context5000.feat100.R500.inf-fullFalse.LTFalse.gen1tr1True.reuse100.E200.step1000.bs8.lr0.001.emb256.hdim512.nhead4.nlayer4.ndevice1/seed0/best.ckpt'`, where hyperparameter configuration is recorded in the directory name

To use our **pretrained FoMo-0D model** (supporting input feature=100), please unzip `ckpt.zip`.

## Test on [ADBench](https://arxiv.org/abs/2206.09426) (Zero-Shot)
We follow [DTE](https://arxiv.org/abs/2305.18593) and run the seeds 0,1,2,3,4 for the zero-shot experiments on ADBench. We use "pre-trained-model-config", "inference-config" below to replace the actual configuration parameters. 
- **zero-shot outlier detection**: (example usage: test seed=0 with quantile transformation) 
  - `CUDA_VISIBLE_DEVICES=0 python zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500 test.seed=0 test.preprocess_transform='quantile'`, 
    
  and everything will be saved under `'results/adbench/pre-trained-model-config/inference-config'`
- **sort results over 5 seeds**: 
  - `python sort_zero_shot_result.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500 test.preprocess_transform='quantile'`,
  
  and everything is sorted and saved as `'results/adbench/pre-trained-model-config/inference-config.csv'`
- **p-value test**: 
  - `python wilcoxon_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500 test.preprocess_transform='quantile'`,
  
  and everything will be saved under `'p_val_test/adbench/pre-trained-model-config/inference-config'`

## Using Pretrained FoMo-0D from HuggingFace

We also host the checkpoints of our best pretrained FoMo-0D model on HuggingFace (thanks for the [suggestion](https://github.com/A-Chicharito-S/FoMo-0D/issues/1) from [Niels](https://github.com/NielsRogge)). Below is a minimal example of how to use it, which requires having a local copy of our repo (to have access to the model definitions and [fomo_hub.py](https://github.com/A-Chicharito-S/FoMo-0D/blob/main/fomo_hub.py)). You can find the HuggingFace page of our model [here](https://huggingface.co/YuchenShen/FoMo-0D).

```python
import torch
from fomo_hub import FoMo0DHub   # the HuggingFace hub-aware class wrapper for FoMo-0D

def main():
    # 1) Download + rebuild the model from the Hub
    repo_id = "YuchenShen/FoMo-0D"
    model = FoMo0DHub.from_pretrained(repo_id, map_location="cpu").eval()

    print("Model loaded successfully from Hub.")

    # 2) Create dummy input matching FoMo-0D’s expected shape (seq_len, batch_size, num_features)
    # For example: (seq_len=10, batch=3, num_features=100)
    bs = 3
    train_x = torch.randn(5000, bs, model.config["num_features"])
    test_x = torch.randn(10, bs, model.config["num_features"])


    # 3) Run forward pass
    with torch.no_grad():
        out = model(train_x=train_x, test_x=test_x)  # (test_x_seq_len=10, batch_size, num_classes=2)

    print(f"Output shape: {out.shape} and type: {type(out)}")
    # would be: "Output shape: torch.Size([10, 3, 2]) and type: <class 'torch.Tensor'>"

if __name__ == "__main__":
    main()
```
**Note** that to deal with data with input features greate or smaller than the `num_features` of FoMo-0D, we follow the procedure of [PFNs](https://arxiv.org/abs/2112.10510) by subsampling or zero-padding to make the input features consistent with FoMo-0D's `num_feature`(=100). We also detail such procedure in **Appendix C.2 Training and inference** of [our paper](https://arxiv.org/abs/2409.05672), and the implementation (with feature transformation) [here](https://github.com/A-Chicharito-S/FoMo-0D/blob/main/zero_shot_on_adbench.py#L212-L226). 

## Citation
```
@article{
shen2025fomod,
title={FoMo-0D: A Foundation Model for Zero-shot Tabular Outlier Detection},
author={Yuchen Shen and Haomin Wen and Leman Akoglu},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=XCQzwpR9jE},
note={}
}
```
