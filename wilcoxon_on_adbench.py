import os.path

import pandas as pd
import re
from scipy.stats import wilcoxon
import hydra
from omegaconf import DictConfig
import numpy as np

dataset_info = {'20news': {'#sample': 11905, '#feature': 768}, 'agnews': {'#sample': 10000, '#feature': 768},
                'aloi': {'#sample': 49534, '#feature': 27}, 'amazon': {'#sample': 10000, '#feature': 768},
                'annthyroid': {'#sample': 7200, '#feature': 6}, 'backdoor': {'#sample': 95329, '#feature': 196},
                'breastw': {'#sample': 683, '#feature': 9}, 'campaign': {'#sample': 41188, '#feature': 62},
                'cardio': {'#sample': 1831, '#feature': 21}, 'cardiotocography': {'#sample': 2114, '#feature': 21},
                'celeba': {'#sample': 202599, '#feature': 39}, 'census': {'#sample': 299285, '#feature': 500},
                'CIFAR10': {'#sample': 5263, '#feature': 512}, 'cover': {'#sample': 286048, '#feature': 10},
                'donors': {'#sample': 619326, '#feature': 10}, 'FashionMNIST': {'#sample': 6315, '#feature': 512},
                'fault': {'#sample': 1941, '#feature': 27}, 'fraud': {'#sample': 284807, '#feature': 29},
                'glass': {'#sample': 214, '#feature': 7}, 'hepatitis': {'#sample': 80, '#feature': 19},
                'http': {'#sample': 567498, '#feature': 3}, 'imdb': {'#sample': 10000, '#feature': 768},
                'internetads': {'#sample': 1966, '#feature': 1555}, 'ionosphere': {'#sample': 351, '#feature': 32},
                'landsat': {'#sample': 6435, '#feature': 36}, 'letter': {'#sample': 1600, '#feature': 32},
                'lymphography': {'#sample': 148, '#feature': 18}, 'magic.gamma': {'#sample': 19020, '#feature': 10},
                'mammography': {'#sample': 11183, '#feature': 6}, 'mnist': {'#sample': 7603, '#feature': 100},
                'MNIST-C': {'#sample': 10000, '#feature': 512}, 'musk': {'#sample': 3062, '#feature': 166},
                'MVTec-AD': {'#sample': 5354, '#feature': 512}, 'optdigits': {'#sample': 5216, '#feature': 64},
                'pageblocks': {'#sample': 5393, '#feature': 10}, 'pendigits': {'#sample': 6870, '#feature': 16},
                'pima': {'#sample': 768, '#feature': 8}, 'satellite': {'#sample': 6435, '#feature': 36},
                'satimage-2': {'#sample': 5803, '#feature': 36}, 'shuttle': {'#sample': 49097, '#feature': 9},
                'skin': {'#sample': 245057, '#feature': 3}, 'smtp': {'#sample': 95156, '#feature': 3},
                'spambase': {'#sample': 4207, '#feature': 57}, 'speech': {'#sample': 3686, '#feature': 400},
                'stamps': {'#sample': 340, '#feature': 9}, 'SVHN': {'#sample': 5208, '#feature': 512},
                'thyroid': {'#sample': 3772, '#feature': 6}, 'vertebral': {'#sample': 240, '#feature': 6},
                'vowels': {'#sample': 1456, '#feature': 12}, 'waveform': {'#sample': 3443, '#feature': 21},
                'wbc': {'#sample': 223, '#feature': 9}, 'wdbc': {'#sample': 367, '#feature': 30},
                'wilt': {'#sample': 4819, '#feature': 5}, 'wine': {'#sample': 129, '#feature': 13},
                'wpbc': {'#sample': 198, '#feature': 33}, 'yeast': {'#sample': 1484, '#feature': 8},
                'yelp': {'#sample': 10000, '#feature': 768}}


# Function to extract mean and std from the "mean(std)" format
def extract_mean_std(entry):
    match = re.match(r"([0-9.]+)\(([0-9.]+)\)", entry)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        print('mean & std are not properly extracted for {}'.format(entry))
        raise BaseException
    # return float('-inf'), None  # Use -inf to denote missing or unparsable values


def get_baseline_topn(setting, metric, baseline, max_feat, n=5, sort_by_p_val=False):
    # Load the Excel/csv file
    if baseline == 'paper':
        file_path = 'DTE/paper_result/{}_{}.xlsx'.format(setting, metric)
        df = pd.read_excel(file_path)
    elif baseline == 'KNN':
        file_path = 'DTE/KNN_result/{}_{}_all.csv'.format(setting, metric.upper())
        df = pd.read_csv(file_path)
    elif baseline == 'DTE_NP':
        file_path = 'DTE/DTE_NP_results/{}_{}_all_dataset.csv'.format(setting, metric.upper())
        df = pd.read_csv(file_path)
    elif baseline == 'DTE_C':
        file_path = 'DTE/DTE_C_results/{}_{}_all_dataset.csv'.format(setting, metric.upper())
        df = pd.read_csv(file_path)
    elif baseline == 'ICL':
        file_path = 'DTE/ICL_results/{}_{}_all_dataset.csv'.format(setting, metric.upper())
        df = pd.read_csv(file_path)
    else:
        raise FileNotFoundError

    method2metric_vals = {}  # method as key, list of metric-val as value

    for index, row in df.iterrows():  # rows are datasets, and columns are methods
        dataset = row.iloc[0]
        for method in df.columns[1:]:
            mean_value, std_value = extract_mean_std(row[method])
            if method not in method2metric_vals.keys():
                method2metric_vals[method] = [mean_value]
            else:
                method2metric_vals[method].append(mean_value)

    if sort_by_p_val:
        method_rank = {}
        for method_i in method2metric_vals.keys():
            method_i_vals = method2metric_vals[method_i]
            method_rank[method_i] = []
            for method_j in method2metric_vals.keys():
                if method_j == method_i:
                    method_rank[method_i].append(-1)
                    continue
                method_j_vals = method2metric_vals[method_j]

                _, p_val_i2j = perform_paired_test(baseline=method_i_vals, ours=method_j_vals, alternative='greater')
                # a small p-val means we accept the alternative, which means method_i is greater than method_j; thus,
                # a stronger method means we prefer a smaller p-val
                method_rank[method_i].append(p_val_i2j)

        method_names = list(method_rank.keys())
        method_data = np.array([p_vals for p_vals in method_rank.values()])

        method_rank = {method: sum(p_vals) for method, p_vals in method_rank.items()}
        topn_method = [k for k, v in sorted(method_rank.items(), key=lambda item: item[1], reverse=False)][:n]
        # reverse=False ---> ascending order, small p-values will be favoured
    else:
        method_data, method_names = None, None
        method_rank = {method: sum(mean) for method, mean in method2metric_vals.items()}
        topn_method = [k for k, v in sorted(method_rank.items(), key=lambda item: item[1], reverse=True)][:n]
        # reverse=True ---> descending order, large means will be favoured

    result_dict = {}

    sorted_topn_dict = get_results_based_on_methods(methods=topn_method, df=df, max_feat=max_feat)
    # {'0': [...], '1': [...], ...}
    result_dict[f'{setting}-{metric}'] = sorted_topn_dict

    return result_dict, topn_method, {metric: {'data': method_data, 'name': method_names}}


def get_results_based_on_methods(methods, df, max_feat):
    topn_results = {i: {} for i in range(len(methods))}

    for index, row in df.iterrows():  # rows are datasets, columns are methods
        dataset = row.iloc[0]

        for i, method in enumerate(methods):
            mean_value, std_value = extract_mean_std(row[method])
            topn_results[i][dataset] = mean_value
    return {i: sort_result(inst_dict=topn_results[i], max_feat=max_feat) for i in topn_results.keys()}


def get_our_result(cfg, setting, metric, max_feat):
    train_cfg = cfg.train
    prior_gmm_cfg = cfg.prior.gmm
    test_cfg = cfg.test
    # train hyperparameters
    seq_len = train_cfg.seq_len
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
    apply_linear_transform = train_cfg.apply_linear_transform
    num_device = train_cfg.num_device
    train_seed = train_cfg.seed
    # prior hyperparameters
    max_feature_dim = prior_gmm_cfg.max_feature_dim
    inflate_full = prior_gmm_cfg.inflate_full
    # test hyperparameters
    feature_truncation = test_cfg.feature_truncation
    preprocess_transform = test_cfg.preprocess_transform
    inf_len = test_cfg.inf_len

    config_details = f'context{seq_len}.feat{max_feature_dim}.R{num_R}.inf-full{inflate_full}.LT{apply_linear_transform}.gen1tr1{gen_one_train_one}.reuse{reuse_data_every_n}.E{epochs}.step{steps_per_epoch}.bs{batch_size}.lr{lr}.emb{emsize}.hdim{nhid}.nhead{nhead}.nlayer{nlayer}.ndevice{num_device}'
    if train_cfg.last_layer_no_R:
        config_details = f'last_layer_no_R{train_cfg.last_layer_no_R}.{config_details}'

    if train_cfg.extra_heading != '':
        config_details = f'{train_cfg.extra_heading}.{config_details}'

    identifier = f'train_seed={train_seed}.transform={preprocess_transform}.feat_truncation={feature_truncation}'
    if test_cfg.extra_suffix != '':
        identifier = identifier + f'.{test_cfg.extra_suffix}'
    path = f'results/adbench/inf{inf_len}.{config_details}/{identifier}'

    file_path = f'{path}_{setting}_all_dataset_mean_std.csv'
    df = pd.read_csv(file_path)
    names = None
    numbers = None
    for column_name, column_data in df.items():
        if column_name == 'dataset':
            names = column_data.tolist()
        if metric.lower() == column_name or column_name in metric.lower():  # second corresponds to 'f1' & 'AUCF1'
            numbers = column_data.tolist()
    numbers = [extract_mean_std(entry=entry)[0] for entry in numbers]
    result = dict(zip(names, numbers))

    return sort_result(inst_dict=result, max_feat=max_feat)


def perform_paired_test(baseline, ours, alternative='greater'):
    res = wilcoxon(baseline, ours, alternative=alternative)
    # when p-value<0.05, we reject the null hypothesis & accept the alternative that baseline is 'greater' than ours
    # (i.e., baseline-ours > 0)
    statistic, p_value = res.statistic, res.pvalue
    return statistic, p_value


def sort_result(inst_dict, max_feat=None):
    # make sure the datasets are in the same order for baselines & ours
    sorted_names = sorted(inst_dict.keys())
    if max_feat is None:
        return [inst_dict[dataset] for dataset in sorted_names]
    else:
        # filter_dim = [dataset for dataset, info in dataset_info.items() if info['#features'] <= max_feat]
        return [inst_dict[dataset] for dataset in sorted_names if dataset_info[dataset]['#feature'] <= max_feat]


def compared_with_topn(ours, topn_results_dict, alternative='greater'):
    p_val_dict = {}
    for config in topn_results_dict.keys():  # e.g., 'semi', 'unsup'
        p_val_dict[config] = []
        sorted_dict = topn_results_dict[config]  # {'0': [...], '1': [...], ...}
        for i in sorted_dict.keys():
            top_i_result = sorted_dict[i]
            _, vs_top_i = perform_paired_test(baseline=top_i_result, ours=ours[config], alternative=alternative)
            p_val_dict[config].append(vs_top_i)

    return p_val_dict


def get_result(cfg, setting, metric, baseline, max_feat, alternative, topn, sort_by_p_val):
    topn_results_dict, topn_method, p_vals_matrix = get_baseline_topn(setting, metric, baseline=baseline,
                                                                      max_feat=max_feat, n=topn,
                                                                      sort_by_p_val=sort_by_p_val)

    our_results = {f'{setting}-{metric}': get_our_result(cfg=cfg, setting=setting, metric=metric, max_feat=max_feat)}

    compared_result = compared_with_topn(ours=our_results, topn_results_dict=topn_results_dict, alternative=alternative)

    p_val_result = {'method': dict(zip(topn_method, topn_method))}
    # first `topn_method` is place-holder, no specific meaning (similar below)

    for config in compared_result.keys():
        p_val_result[f'{cfg.train.seq_len}-{config}'] = dict(zip(topn_method, compared_result[config]))

    return p_val_result, p_vals_matrix  # {metric: {'data': method_data, 'name': method_names}}


@hydra.main(version_base='1.3', config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    train_cfg = cfg.train
    prior_gmm_cfg = cfg.prior.gmm
    test_cfg = cfg.test
    # train hyperparameters
    seq_len = train_cfg.seq_len
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
    apply_linear_transform = train_cfg.apply_linear_transform
    num_device = train_cfg.num_device
    train_seed = train_cfg.seed
    # prior hyperparameters
    max_feature_dim = prior_gmm_cfg.max_feature_dim
    inflate_full = prior_gmm_cfg.inflate_full
    # test hyperparameters
    feature_truncation = test_cfg.feature_truncation
    preprocess_transform = test_cfg.preprocess_transform
    inf_len = test_cfg.inf_len

    config_details = f'context{seq_len}.feat{max_feature_dim}.R{num_R}.inf-full{inflate_full}.LT{apply_linear_transform}.gen1tr1{gen_one_train_one}.reuse{reuse_data_every_n}.E{epochs}.step{steps_per_epoch}.bs{batch_size}.lr{lr}.emb{emsize}.hdim{nhid}.nhead{nhead}.nlayer{nlayer}.ndevice{num_device}'
    if train_cfg.last_layer_no_R:
        config_details = f'last_layer_no_R{train_cfg.last_layer_no_R}.{config_details}'

    if train_cfg.extra_heading != '':
        config_details = f'{train_cfg.extra_heading}.{config_details}'

    identifier = f'train_seed={train_seed}.transform={preprocess_transform}.feat_truncation={feature_truncation}'
    if test_cfg.extra_suffix != '':
        identifier = identifier + f'.{test_cfg.extra_suffix}'
    path = f'p_val_test/adbench/inf{inf_len}.{config_details}/{identifier}'

    for setting in ['semi']:
        for metric in ['aucroc', 'aucpr', 'aucf1']:
            if not os.path.exists(f'{path}/{metric}'):
                os.makedirs(f'{path}/{metric}')
            for (baseline, topn) in [('paper', 26), ('KNN', 6), ('DTE_NP', 6), ('DTE_C', 6), ('ICL', 6)]:
                file_path = f'{path}/{metric}/{metric}_vs{baseline}_top{topn}.csv'
                for max_feat in [None, 20, 50, 100, 200, 500]:
                    p_val_result, _ = get_result(cfg=cfg, setting=setting, metric=metric, baseline=baseline,
                                                 max_feat=max_feat, alternative='greater', topn=topn,
                                                 sort_by_p_val=True)
                    df = pd.DataFrame(p_val_result)

                    # Transpose the DataFrame to match the desired structure
                    df = df.transpose().reset_index()
                    df.columns = ['max_feat_{}'.format(max_feat)] + ['vs. top{}'.format(i + 1) for i in range(topn)]

                    # Save the DataFrame to a CSV file
                    df.to_csv(file_path, mode='a', index=False)


if __name__ == "__main__":
    main()
