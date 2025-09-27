import os.path

try:
    import pfns
except ImportError:
    raise ImportError("Please restart runtime by i) clicking on \'Runtime\' and then ii) clicking \'Restart runtime\'")

import pickle
from pytorch_lightning.strategies import DDPStrategy
import torch
import time
import random
from torch import nn
from pfns import encoders
import hydra
from omegaconf import DictConfig
from data_prior.parallel_generator_torch import PriorTrainDataGenerator
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pfns.train_parallel_torch import ZeroShotOD
import numpy as np
import wandb


class single_eval_pos_generator:
    def __init__(self, mode, num_test_x, seq_len, num_R):
        self.mode = mode
        self.num_test_x = num_test_x
        self.seq_len = seq_len
        self.single_eval_pos = None
        self.num_R = 0 if num_R is None else num_R

        if self.mode == 'constant':
            assert self.num_test_x > self.num_R, \
                print(f'we are in constant mode, please make sure '
                      f'seq-len{self.seq_len}-num_test_x{self.num_test_x}+num_R{self.num_R}<seq-len{self.seq_len}')

    def generate(self):
        if self.mode == 'constant':
            self.single_eval_pos = self.seq_len - self.num_test_x + self.num_R
        else:
            self.single_eval_pos = random.choices(range(0, self.seq_len - self.num_R))[0] + self.num_R
        # single_eval_pos >= num_R
        return self.single_eval_pos


def make_pl_model(cfg, get_batch_function, seq_len, num_features, hps,
                  generator_mode='constant', num_class=2, num_R=None,
                  model_para_dict=None, train_extra_dict=None, resume_from_ckpt=False):
    criterion = nn.CrossEntropyLoss(weight=torch.ones(size=(num_class,)) / num_class, reduction='none',
                                    ignore_index=hps['ignore_index'])

    single_eval_pos_gen = single_eval_pos_generator(mode=generator_mode, num_test_x=hps['num_test_x'], seq_len=seq_len,
                                                    num_R=num_R)
    # now train
    pl_model = ZeroShotOD(cfg=cfg,  # the prior is the key. It defines what we train on.
                          priordataloader_class_or_get_batch=get_batch_function, criterion=criterion,
                          # how to encode the x and y inputs to the transformer
                          encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
                          y_encoder_generator=encoders.Linear,
                          # these are given to the prior, which needs to know how many features we have etc
                          extra_prior_kwargs_dict={'num_features': num_features,
                                                   'hyperparameters': hps,
                                                   'pt_dataloader': {'num_workers': 0, 'pin_memory': True},
                                                   'num_R': num_R},
                          single_eval_pos_gen=single_eval_pos_gen,
                          # ---> control the training / test sample size
                          progress_bar=True,
                          train_extra_dict=train_extra_dict,
                          resume_from_ckpt=resume_from_ckpt,
                          model_para_dict=model_para_dict,  # input as **model_extra_args

                          )
    return pl_model


def set_seed(seed: int = 42) -> None:
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# CUDA_VISIBLE_DEVICES=1 python pretrain.py
@hydra.main(version_base='1.3', config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    prior_train_data_gen = PriorTrainDataGenerator(cfg=cfg)

    train_cfg = cfg.train
    prior_gmm_cfg = cfg.prior.gmm
    # train hyperparameters
    seq_len = train_cfg.seq_len
    hyperparameters = train_cfg.hyperparameters
    batch_size = train_cfg.batch_size
    epochs = train_cfg.epochs
    steps_per_epoch = train_cfg.steps_per_epoch
    lr = train_cfg.lr
    emsize = train_cfg.emsize
    nhead = train_cfg.nhead
    nhid = train_cfg.nhid
    nlayer = train_cfg.nlayer
    num_R = train_cfg.num_R
    reuse_data_every_n = train_cfg.reuse_data_every_n
    gen_one_train_one = train_cfg.gen_one_train_one
    resume_from_ckpt = train_cfg.resume_from_ckpt
    apply_linear_transform = train_cfg.apply_linear_transform
    seed = train_cfg.seed
    num_device = train_cfg.num_device
    # prior hyperparameters
    max_feature_dim = prior_gmm_cfg.max_feature_dim
    inflate_full = prior_gmm_cfg.inflate_full

    set_seed(seed=seed)

    generator_mode = hyperparameters['mode']

    config_details = f'context{seq_len}.feat{max_feature_dim}.R{num_R}.inf-full{inflate_full}.LT{apply_linear_transform}.gen1tr1{gen_one_train_one}.reuse{reuse_data_every_n}.E{epochs}.step{steps_per_epoch}.bs{batch_size}.lr{lr}.emb{emsize}.hdim{nhid}.nhead{nhead}.nlayer{nlayer}.ndevice{num_device}'
    if train_cfg.last_layer_no_R:
        config_details = f'last_layer_no_R{train_cfg.last_layer_no_R}.{config_details}'

    if train_cfg.extra_heading != '':
        config_details = f'{train_cfg.extra_heading}.{config_details}'

    save_path = f'{train_cfg.model_dir}/{config_details}/seed{seed}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start_time = time.time()

    zero_shot_od_pl_model = make_pl_model(cfg=cfg,
                                          get_batch_function=prior_train_data_gen.get_batch_for_NdMclusterGaussian,
                                          seq_len=seq_len, num_features=max_feature_dim, num_class=2,
                                          generator_mode=generator_mode, hps=hyperparameters,
                                          num_R=num_R,
                                          model_para_dict={'num_R': num_R,
                                                           'last_layer_no_R': train_cfg.last_layer_no_R},
                                          train_extra_dict=None if not gen_one_train_one else {
                                                     'prior_train_data_gen': prior_train_data_gen},
                                          resume_from_ckpt=resume_from_ckpt)

    # Initialize the WandbLogger
    if train_cfg.logging:
        logger = WandbLogger(project='FoMo_0D', name=config_details)
        wandb.login(key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    else:
        logger = None

    ckpt_callbacks = []
    # Define the checkpoint callback
    train_ckpt_callback = ModelCheckpoint(
        monitor='train_loss',  # The metric to monitor
        mode='min',  # 'min' because you want to save the model with the smallest validation loss
        save_top_k=1,  # Save only the best model
        dirpath=save_path,
        filename='best',
        # Name of the saved model file
        verbose=True,  # Print information about saving
        save_last=True
    )
    ckpt_callbacks.append(train_ckpt_callback)

    if train_cfg.use_validation:
        val_ckpt_callback = ModelCheckpoint(
            monitor='val_loss',  # The metric to monitor
            mode='min',  # 'min' because you want to save the model with the smallest validation loss
            save_top_k=1,  # Save only the best model
            dirpath=save_path,
            filename='best_val',
            # Name of the saved model file
            verbose=True  # Print information about saving
        )
        ckpt_callbacks.append(val_ckpt_callback)

    print(f'setting the max epochs to {epochs}')
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=ckpt_callbacks,
        logger=logger,  # Use the WandbLogger
        max_epochs=epochs,
        enable_progress_bar=True,
        limit_val_batches=None if train_cfg.use_validation else 0,
        check_val_every_n_epoch=1,
        # accumulate_grad_batches=1,  # gradient accumulation
        # precision=16,  # Enable mixed precision training
        devices=num_device,
        reload_dataloaders_every_n_epochs=1  # if gen_one_train_one else 0
    )

    trainer.fit(zero_shot_od_pl_model, ckpt_path=f'{save_path}/last.ckpt' if resume_from_ckpt else None)

    # visualize train/val dynamics
    train_time = time.time() - start_time
    train_loss, val_loss = zero_shot_od_pl_model.train_losses, zero_shot_od_pl_model.val_losses
    val_loss = val_loss[-len(train_loss):]  # val loss might contain the sanity check validation loss
    with open(
            save_path + '/train_val_loss.pickle', 'wb') as handle:
        pickle.dump({'train_loss': train_loss, 'val_loss': val_loss, 'train_time': train_time}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print('total training time: {}'.format(train_time / 60))


if __name__ == "__main__":
    main()
