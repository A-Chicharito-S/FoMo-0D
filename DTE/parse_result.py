import argparse
import numpy as np

import os
import pandas as pd

import time

from adbench.myutils import Utils

from data_generator import DataGenerator


def low_density_anomalies(test_log_probs, num_anomalies):
    """ Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob
    """
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies - 1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds


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


def parse_one_csv(seed, metric):
    utils = Utils()  # utils function
    utils.set_seed(seed)

    file_path = f'results/semi/{seed}_{metric}.csv'
    df = pd.read_csv(file_path)

    # Get column names
    column_names = df.columns.tolist()[1:]
    dataset2num_test = get_dataset2num_test(seed=seed)

    dataset_dict = {}
    for dataset in datasets_list:
        for _, row in df.iterrows():
            row = row.tolist()
            dataset_name = row[0]
            data = row[1:]

            if dataset not in dataset_name:
                continue

            if dataset not in dataset_dict.keys():
                dataset_dict[dataset] = []
            if 'InferenceTime' in metric:
                dataset_dict[dataset].append(np.array(data) / dataset2num_test[dataset_name])
            else:
                dataset_dict[dataset].append(np.array(data)*100)  # AUCPR, AUCROC

    for dataset in dataset_dict.keys():
        dataset_dict[dataset] = np.array(dataset_dict[dataset])

        dataset_dict[dataset] = np.mean(dataset_dict[dataset], axis=0).tolist()

    return dataset_dict, column_names


def get_dataset2num_test(seed):
    datagenerator = DataGenerator(seed=seed, test_size=0.5, normal=True)  # data generator

    utils = Utils()  # utils function
    utils.set_seed(seed)

    # Get the datasets from ADBench
    dataset2num_test = {}
    for dataset_list in [datagenerator.dataset_list_classical, datagenerator.dataset_list_cv,
                         datagenerator.dataset_list_nlp]:
        for dataset in dataset_list:
            '''
            la: ratio of labeled anomalies, from 0.0 to 1.0
            realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
            noise_type: inject data noises for testing prior robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
            '''
            # print(dataset)
            # if dataset in df_AUCROC.index.values:
            #     continue

            # import the dataset
            datagenerator.dataset = dataset  # specify the dataset name
            data = datagenerator.generator(la=0, max_size=50000)  # maximum of 50,000 data points are available

            dataset2num_test[dataset] = len(data['y_test'])

    return dataset2num_test


def main():
    for metric in ['AUCPR', 'AUCROC', 'InferenceTime']:
        column_names = None
        result_over_seed = {dataset: [] for dataset in datasets_list}
        for seed in [0, 1, 2, 3, 4]:
            dataset_dict, column_names = parse_one_csv(seed=seed, metric=metric)

            # update the results
            for dataset in dataset_dict.keys():
                result_over_seed[dataset].append(dataset_dict[dataset])

        final_result = {}
        for dataset in result_over_seed.keys():
            result = np.array(result_over_seed[dataset])  # num_seed, num_model
            mean = np.mean(result, axis=0).tolist()
            std = np.std(result, axis=0).tolist()
            if metric == 'InferenceTime':
                final_result[header_mapping[dataset]] = {column_names[i]: '{:.5f}'.format(mean[i]) for i in range(len(column_names))}
            else:
                final_result[header_mapping[dataset]] = {column_names[i]: '{:.2f}({:.2f})'.format(mean[i], std[i]) for i in
                                                         range(len(column_names))}

        df = pd.DataFrame(final_result)

        # Transpose the DataFrame to match the desired structure
        df = df.transpose().reset_index()
        df.columns = ['dataset'] + column_names

        # Save the DataFrame to a CSV file
        df.to_csv('results/semi_{}_all.csv'.format(metric), mode='a', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--setting', type=str,
                        default='semi', help='choice of experimental setting (semi or unsup)')
    parser.add_argument('--seed', type=int,
                        default=42, help='random seed')

    dataset_stat = {}
    args = parser.parse_args()
    datasets_list = ['agnews', '20news', 'SVHN', 'FashionMNIST', 'MNIST-C', 'CIFAR10', 'MVTec-AD', '44_Wilt',
                     '24_mnist', '15_Hepatitis', 'yelp', '35_SpamBase', '34_smtp', '36_speech', '29_Pima',
                     '17_InternetAds', '37_Stamps', '7_Cardiotocography', 'imdb', '1_ALOI', '46_WPBC', '47_yeast',
                     '13_fraud', '25_musk', '12_fault', 'amazon', '5_campaign', '4_breastw', '14_glass', '38_thyroid',
                     '16_http', '26_optdigits', '33_skin', '3_backdoor', '20_letter', '39_vertebral', '31_satimage-2',
                     '43_WDBC', '8_celeba', '19_landsat', '40_vowels', '32_shuttle', '45_wine', '2_annthyroid',
                     '9_census', '41_Waveform', '6_cardio', '42_WBC', '18_Ionosphere', '10_cover', '27_PageBlocks',
                     '28_pendigits', '30_satellite', '22_magic.gamma', '11_donors', '21_Lymphography', '23_mammography']

    main()
