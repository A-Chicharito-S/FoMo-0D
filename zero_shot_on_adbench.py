import os

from DTE.data_generator import DataGenerator
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
import os.path

try:
    import pfns
except ImportError:
    raise ImportError("Please restart runtime by i) clicking on \'Runtime\' and then ii) clicking \'Restart runtime\'")

import pandas as pd

import torch

import numpy as np
import random
from torch import nn
from pfns.train_parallel import make_model_od
from pfns import encoders
import time
from adbench.myutils import Utils
from data_prior.feature_transform import FeatureTransform
import sklearn.metrics as skm


def make_model(seq_len, num_features, hps, emsize, nhead, nhid, nlayers, num_class=2, model_para_dict=None):
    criterion = nn.CrossEntropyLoss(weight=torch.ones(size=(num_class,)) / num_class, reduction='none',
                                    ignore_index=hps['ignore_index'])

    # now train
    model = make_model_od(
        criterion=criterion, encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
        # define the transformer size
        emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers,
        # seq_len defines the size of your datasets (including the test set)
        seq_len=seq_len,
        y_encoder_generator=encoders.Linear,
        # these are given to the prior, which needs to know how many features we have etc
        extra_prior_kwargs_dict={'num_features': num_features},
        model_para_dict=model_para_dict,
    )
    return model


def get_results(model, train_x, test_x, label, save_path, inst_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # train_x: (seq_len-1, 1, d)
    train_y = None
    # test_x: (1[seq-len], num_text_x[batch_size], d[#feat])

    total_time = 0
    with torch.no_grad():
        model.eval()
        # we add our batch dimension, as our transformer always expects that
        logits = []
        from tqdm import tqdm
        for i in tqdm(range(test_x.shape[1])):
            start_time = time.time()
            inst_logits = model(train_x, train_y, test_x[:, i, :].unsqueeze(1))
            # (bs=1, num_test_x=1, num_classes=2)

            duration = time.time() - start_time
            total_time += duration

            inst_logits = inst_logits.squeeze(0)  # squeeze the batch_size=1, get: (num_test_x=1, num_classes=2)
            logits.append(inst_logits)

        logits = torch.concatenate(logits, dim=0).cpu().numpy()

        return logits, total_time


def low_density_anomalies(test_log_probs, num_anomalies):
    """ Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob
    """
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies - 1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds


@hydra.main(version_base='1.3', config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    # feature transform
    FT = FeatureTransform(cfg=cfg)

    train_cfg = cfg.train
    prior_gmm_cfg = cfg.prior.gmm
    test_cfg = cfg.test
    # train hyperparameters
    seq_len = train_cfg.seq_len
    hyperparameters = train_cfg.hyperparameters
    batch_size = train_cfg.batch_size
    epochs = train_cfg.epochs
    steps_per_epoch = train_cfg.steps_per_epoch
    lr = train_cfg.lr
    device = train_cfg.device
    emsize = train_cfg.emsize
    nhead = train_cfg.nhead
    nhid = train_cfg.nhid
    nlayer = train_cfg.nlayer
    num_R = train_cfg.num_R
    reuse_data_every_n = train_cfg.reuse_data_every_n
    gen_one_train_one = train_cfg.gen_one_train_one
    apply_linear_transform = train_cfg.apply_linear_transform
    num_device = train_cfg.num_device
    train_seed = train_cfg.seed
    # prior hyperparameters
    max_feature_dim = prior_gmm_cfg.max_feature_dim
    inflate_full = prior_gmm_cfg.inflate_full
    # test hyperparameters
    feature_truncation = test_cfg.feature_truncation
    seed = test_cfg.seed
    inf_len = test_cfg.inf_len
    preprocess_transform = test_cfg.preprocess_transform

    print(f'testing with seed={seed}--num_R={num_R}--train_seed={train_seed}')

    config_details = f'context{seq_len}.feat{max_feature_dim}.R{num_R}.inf-full{inflate_full}.LT{apply_linear_transform}.gen1tr1{gen_one_train_one}.reuse{reuse_data_every_n}.E{epochs}.step{steps_per_epoch}.bs{batch_size}.lr{lr}.emb{emsize}.hdim{nhid}.nhead{nhead}.nlayer{nlayer}.ndevice{num_device}'
    if train_cfg.last_layer_no_R:
        config_details = f'last_layer_no_R{train_cfg.last_layer_no_R}.{config_details}'

    if train_cfg.extra_heading != '':
        config_details = f'{train_cfg.extra_heading}.{config_details}'

    model_path = f'{train_cfg.model_dir}/{config_details}/seed{train_seed}'

    if not os.path.exists(model_path):
        print('file not found:')
        print(model_path)
        raise FileNotFoundError
    # seq_len, num_features, hps, emsize, nhead, nhid, nlayers, num_class=2, model_para_dict=None
    trained_model = make_model(seq_len=seq_len, num_features=max_feature_dim, hps=hyperparameters,
                               emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayer,
                               num_class=2, model_para_dict={'num_R': num_R})
    trained_model = trained_model.to(device)

    ckpt_path = f'{model_path}/best.ckpt'
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract the state_dict
    state_dict = checkpoint['state_dict']

    # Remove the 'model.' prefix from the keys (if any) saved by pytorch_lightning
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('model.', '')
        new_state_dict[new_key] = state_dict[key]
    trained_model.load_state_dict(new_state_dict)

    print(f'loading from {ckpt_path}')

    print(
        f"Using a Transformer with {sum(p.numel() for p in trained_model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    setting = 'semi'
    utils = Utils()  # utils function
    utils.set_seed(seed)

    if "semi" in setting:
        datagenerator = DataGenerator(seed=seed, test_size=0.5, normal=True)  # data generator
    else:
        datagenerator = DataGenerator(seed=seed, test_size=0, normal=False)  # data generator
    utils = Utils()  # utils function
    utils.set_seed(seed)
    # Get the datasets from ADBench

    for dataset_list in [datagenerator.dataset_list_classical, datagenerator.dataset_list_cv,
                         datagenerator.dataset_list_nlp]:
        for dataset in dataset_list:
            '''
            la: ratio of labeled anomalies, from 0.0 to 1.0
            realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
            noise_type: inject data noises for testing prior robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
            '''
            print(dataset)
            # if dataset in df_AUCROC.index.values:
            #     continue

            identifier = f'train_seed={train_seed}.transform={preprocess_transform}.feat_truncation={feature_truncation}'
            if test_cfg.extra_suffix != '':
                identifier = identifier + f'.{test_cfg.extra_suffix}'

            save_path = f'results/adbench/inf{inf_len}.{config_details}/{identifier}/{dataset}/{setting}'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # import the dataset
            datagenerator.dataset = dataset  # specify the dataset name
            data = datagenerator.generator(la=0, max_size=50000)  # maximum of 50,000 data points are available

            if "unsup" in setting:
                X = data['X_test']
                y = data['y_test']

                indices = np.arange(len(X))
                subset = np.random.choice(indices, size=len(indices), replace=True)

                data = {}
                data['X_train'] = X[subset]
                data['y_train'] = y[subset]

                data['X_test'] = X
                data['y_test'] = y

            def add_feature_transform(x, eval_position):
                if preprocess_transform is None:
                    feature_dim = x.shape[-1]
                    if feature_dim > max_feature_dim:
                        if feature_truncation == 'projection':
                            x = FT.feature_sparse_projection(x=x, num_feature=feature_dim)
                        else:
                            x = FT.feature_subsampling(x=x, num_feature=feature_dim)
                    if feature_dim < max_feature_dim:
                        x = FT.feature_padding(x=x, num_feature=feature_dim)
                else:
                    x = FT.pfn_inference_transform(eval_xs=x, preprocess_transform=preprocess_transform,
                                                   eval_position=eval_position,
                                                   normalize_with_test=True, rescale_with_sqrt=False)
                return x

            def make_train_test(data):
                x_train = data['X_train']
                y_train = data['y_train']

                x_test = data['X_test']
                y_test = data['y_test']

                if x_train.shape[0] <= inf_len - 1:
                    train_x = x_train
                else:
                    train_sub_indices = np.random.choice(x_train.shape[0], inf_len - 1, replace=False)
                    train_x = x_train[train_sub_indices]

                train_and_test = add_feature_transform(x=np.concatenate([train_x, x_test], axis=0),
                                                       eval_position=len(train_x))

                train_x, x_test = train_and_test[:len(train_x), :], train_and_test[len(train_x):, :]

                test_in = x_test[y_test == 0]  # #inst, d
                test_out = x_test[y_test == 1]  # #inst, d

                train_x = torch.from_numpy(train_x).to(device).unsqueeze(1).float()
                test_in = torch.from_numpy(test_in).to(device).unsqueeze(0).float()
                test_out = torch.from_numpy(test_out).to(device).unsqueeze(0).float()

                print('train_x shape:', train_x.shape)

                # train_x: (seq_len-1, 1, d)
                # test_x: (1[seq-len], num_text_x[batch_size], d[#feat])
                return train_x, test_in, test_out

            train_x, test_in, test_out = make_train_test(data)
            logits_in, time_in = get_results(model=trained_model, train_x=train_x, test_x=test_in, label=0,
                                             save_path=save_path + '/seed{}'.format(seed), inst_type='in')
            logits_la, time_la = get_results(model=trained_model, train_x=train_x, test_x=test_out, label=1,
                                             save_path=save_path + '/seed{}'.format(seed), inst_type='la')

            logits = np.concatenate([logits_in, logits_la], axis=0)  # #samples, 2
            labels = np.array([0] * len(logits_in) + [1] * len(logits_la))

            exp_logits = np.exp(logits)
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Use the probability of the positive class
            score = probabilities[:, 1]

            # following DTE:
            indices = np.arange(len(labels))
            p = low_density_anomalies(-score, len(indices[labels == 1]))
            f1_score = skm.f1_score(labels, p)

            inds = np.where(np.isnan(score))
            score[inds] = 0
            result = utils.metric(y_true=labels, y_score=score)

            result = {seed: {'aucroc': result['aucroc'], 'aucpr': result['aucpr'], 'f1': f1_score,
                             'time-per-inst': (time_in + time_la) / len(labels),
                             'context-len': train_x.shape[0]}}

            df = pd.DataFrame(result)

            # Transpose the DataFrame to match the desired structure
            df = df.transpose().reset_index()
            df.columns = ['seed', 'aucroc', 'aucpr', 'f1', 'time-per-inst', 'context-len']

            # Save the DataFrame to a CSV file
            df.to_csv(save_path + '/result.csv', mode='a', index=False)


def run_with_overrides(base_overrides, overrides):
    GlobalHydra.instance().clear()
    initialize(config_path="configuration")
    cfg = compose(config_name="config", overrides=base_overrides + overrides)
    main(cfg)


if __name__ == "__main__":
    override_configs = [[f"test.seed={0}"], [f"test.seed={1}"], [f"test.seed={2}"], [f"test.seed={3}"],
                        [f"test.seed={4}"]]
    import sys
    # Collect command-line arguments for Hydra overrides
    base_overrides = sys.argv[1:]

    for overrides in override_configs:
        run_with_overrides(base_overrides, overrides)
    # main()