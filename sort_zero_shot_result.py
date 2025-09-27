import hydra
from omegaconf import DictConfig
import os.path
import numpy as np
import os
import pandas as pd

header_mapping = {
    "1_ALOI": "aloi",
    "amazon": "amazon",
    "2_annthyroid": "annthyroid",
    "3_backdoor": "backdoor",
    "4_breastw": "breastw",
    "5_campaign": "campaign",
    "6_cardio": "cardio",
    "7_Cardiotocography": "cardiotocography",
    "8_celeba": "celeba",
    "9_census": "census",
    "10_cover": "cover",
    "11_donors": "donors",
    "12_fault": "fault",
    "13_fraud": "fraud",
    "14_glass": "glass",
    "15_Hepatitis": "hepatitis",
    "16_http": "http",
    "imdb": "imdb",
    "17_InternetAds": "internetads",
    "18_Ionosphere": "ionosphere",
    "19_landsat": "landsat",
    "20_letter": "letter",
    "21_Lymphography": "lymphography",
    "22_magic.gamma": "magic.gamma",
    "23_mammography": "mammography",
    "24_mnist": "mnist",
    "25_musk": "musk",
    "26_optdigits": "optdigits",
    "27_PageBlocks": "pageblocks",
    "28_pendigits": "pendigits",
    "29_Pima": "pima",
    "30_satellite": "satellite",
    "31_satimage-2": "satimage-2",
    "32_shuttle": "shuttle",
    "33_skin": "skin",
    "34_smtp": "smtp",
    "35_SpamBase": "spambase",
    "36_speech": "speech",
    "37_Stamps": "stamps",
    "38_thyroid": "thyroid",
    "39_vertebral": "vertebral",
    "40_vowels": "vowels",
    "41_Waveform": "waveform",
    "42_WBC": "wbc",
    "43_WDBC": "wdbc",
    "44_Wilt": "wilt",
    "45_wine": "wine",
    "46_WPBC": "wpbc",
    "47_yeast": "yeast",
    "yelp": "yelp",
    "MNIST-C": "MNIST-C",
    "FashionMNIST": "FashionMNIST",
    "CIFAR10": "CIFAR10",
    "SVHN": "SVHN",
    "MVTec-AD": "MVTec-AD",
    "20news": "20news",
    "agnews": "agnews"
}

datasets_list = ['agnews', '20news', 'SVHN', 'FashionMNIST', 'MNIST-C', 'CIFAR10', 'MVTec-AD', '44_Wilt',
                 '24_mnist', '15_Hepatitis', 'yelp', '35_SpamBase', '34_smtp', '36_speech', '29_Pima',
                 '17_InternetAds', '37_Stamps', '7_Cardiotocography', 'imdb', '1_ALOI', '46_WPBC', '47_yeast',
                 '13_fraud', '25_musk', '12_fault', 'amazon', '5_campaign', '4_breastw', '14_glass', '38_thyroid',
                 '16_http', '26_optdigits', '33_skin', '3_backdoor', '20_letter', '39_vertebral', '31_satimage-2',
                 '43_WDBC', '8_celeba', '19_landsat', '40_vowels', '32_shuttle', '45_wine', '2_annthyroid',
                 '9_census', '41_Waveform', '6_cardio', '42_WBC', '18_Ionosphere', '10_cover', '27_PageBlocks',
                 '28_pendigits', '30_satellite', '22_magic.gamma', '11_donors', '21_Lymphography', '23_mammography']


@hydra.main(version_base='1.3', config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    setting = 'semi'

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
    reuse_data_every_n = train_cfg.reuse_data_every_n
    gen_one_train_one = train_cfg.gen_one_train_one
    apply_linear_transform = train_cfg.apply_linear_transform
    num_device = train_cfg.num_device
    train_seed = train_cfg.seed
    num_R = train_cfg.num_R
    # prior hyperparameters
    max_feature_dim = prior_gmm_cfg.max_feature_dim
    inflate_full = prior_gmm_cfg.inflate_full
    # test config
    preprocess_transform = test_cfg.preprocess_transform
    feature_truncation = test_cfg.feature_truncation
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

    for dataset in datasets_list:  # for a specific dataset out of the 57 adbench datasets
        dataset_dict = {}
        for dataset_dir in os.listdir(path):
            if dataset not in dataset_dir:  # see whether dataset is a prefix/suffix of the datasets under `path`
                # (since some big datasets are split into different pieces, e.g., MNIST-1/2/...)
                continue

            file_path = f'{path}/{dataset_dir}/{setting}/result.csv'
            df = pd.read_csv(file_path)
            even_rows_df = df.iloc[::2].to_dict(orient='records')

            assert len(even_rows_df) == 5  # 5 seeds
            for inst in even_rows_df:
                seed = float(inst['seed'])
                aucroc = float(inst['aucroc'])
                aucpr = float(inst['aucpr'])
                f1 = float(inst['f1'])
                time_per_inst = float(inst['time-per-inst'])
                context_len = float(inst['context-len'])

                if seed not in dataset_dict.keys():  # create for the first time
                    dataset_dict[seed] = {'aucroc': [aucroc], 'aucpr': [aucpr], 'f1': [f1],
                                          'time-per-inst': [time_per_inst], 'context-len': [context_len]}
                else:
                    dataset_dict[seed]['aucroc'].append(aucroc)
                    dataset_dict[seed]['aucpr'].append(aucpr)
                    dataset_dict[seed]['f1'].append(f1)
                    dataset_dict[seed]['time-per-inst'].append(time_per_inst)
                    dataset_dict[seed]['context-len'].append(context_len)
                    # for those datasets are split into many sub-parts, we first get the avr results for those
                    # sub-parts on one specific seed
        all_aucroc = []
        all_aucpr = []
        all_f1 = []
        all_time_per_inst = []
        all_context_len = []
        if len(dataset_dict[0]['aucroc']) > 1:
            print(dataset, len(dataset_dict[0]['aucroc']))  # this dataset has multiple splits during testing
        for seed in dataset_dict.keys():
            aucroc = sum(dataset_dict[seed]['aucroc']) / len(dataset_dict[seed]['aucroc'])
            aucpr = sum(dataset_dict[seed]['aucpr']) / len(dataset_dict[seed]['aucpr'])
            f1 = sum(dataset_dict[seed]['f1']) / len(dataset_dict[seed]['f1'])

            time_per_inst = sum(dataset_dict[seed]['time-per-inst']) / len(dataset_dict[seed]['time-per-inst'])
            context_len = sum(dataset_dict[seed]['context-len']) / len(dataset_dict[seed]['context-len'])

            all_aucroc.append(aucroc)
            all_aucpr.append(aucpr)
            all_f1.append(f1)

            all_time_per_inst.append(time_per_inst)
            all_context_len.append(context_len)

        all_aucroc = np.array(all_aucroc) * 100
        all_aucpr = np.array(all_aucpr) * 100
        all_f1 = np.array(all_f1) * 100

        all_time_per_inst = np.array(all_time_per_inst)
        all_context_len = np.array(all_context_len)

        mean_aucroc = np.mean(all_aucroc)
        std_aucroc = np.std(all_aucroc)

        mean_time_per_inst = np.mean(all_time_per_inst)
        std_time_per_inst = np.std(all_time_per_inst)

        mean_context_len = np.mean(all_context_len)
        std_context_len = np.std(all_context_len)

        mean_aucpr = np.mean(all_aucpr)
        std_aucpr = np.std(all_aucpr)

        mean_f1 = np.mean(all_f1)
        std_f1 = np.std(all_f1)

        result = {header_mapping[dataset]: {'aucroc': '{:.2f}({:.2f})'.format(mean_aucroc, std_aucroc),
                                            'aucpr': '{:.2f}({:.2f})'.format(mean_aucpr, std_aucpr),
                                            'f1': '{:.2f}({:.2f})'.format(mean_f1, std_f1),
                                            'time-per-inst': '{:.5f}'.format(mean_time_per_inst),
                                            'context-len': '{:.5f}({:.5f})'.format(mean_context_len, std_context_len)}
                  }

        df = pd.DataFrame(result)

        # Transpose the DataFrame to match the desired structure
        df = df.transpose().reset_index()
        df.columns = ['dataset', 'aucroc', 'aucpr', 'f1', 'time-per-inst', 'context-len']
        save_path = f'{path}_{setting}_all_dataset_mean_std.csv'
        file_exists = os.path.isfile(save_path)
        write_header = not file_exists or os.stat(save_path).st_size == 0
        # Save the DataFrame to a CSV file
        df.to_csv(save_path, mode='a', index=False, header=write_header)


if __name__ == "__main__":
    main()
