"""
This file runs the main experiment (in the case of tabular data).
"""

import argparse
import data_preprocessing_tabular
from models.heckman import train_eval_heckman, Heckman_FA
from models.linear_regression import naive_linear_regression
import torch

# Set of pre-processing functions for datasets
dataprep = {
    "compas": data_preprocessing_tabular.preprocess_compas,
    "crime": data_preprocessing_tabular.preprocess_crime,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '--num_exec',
        type=int,
        default=1,
        help="Number of times to execute experiment"
    )
    parser.add_argument(
        '--prediction_assignment',
        type=int,
        default=1,
        help="0: Use default prediction feats; 1: Heckman-FA; 2: Heckman-FA*"
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.7,
        help="Train-test split of p train and (1-p) test"
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        help="GPU ID"
    )
    parser.add_argument(
        '--c',
        type=float,
        default=0.75,
        help="Specify c for the initialization of pi"
    )
    parser.add_argument(
        '--T',
        type=int,
        default=10,
        help="Specify max number of epochs to train modified Greene's method on"
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.75,
        help="Specify learning rate to train modified Greene's method with"
    )
    parser.add_argument(
        '--B',
        type=int,
        default=1000,
        help="Specify number of Gumbel-Softmax samples to draw during assignment extraction"
    )
    parser.add_argument(
        '--num_tests',
        type=int,
        default=1,
        help="Specify number of tests to run"
    )
    parser.add_argument(
        '--rho_range',
        nargs='+',
        required=True,
        help="Specify rho min and rho max"
    )
    args = parser.parse_args()
    dataset = args.dataset
    num_exec = args.num_exec
    psi = args.prediction_assignment
    split = args.split
    gpu_id = args.gpu_id
    c = args.c
    max_epochs = args.T
    lr = args.alpha
    B = args.B
    num_tests = args.num_tests
    rho_min = float(args.rho_range[0])
    rho_max = float(args.rho_range[1])

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU-{}".format(gpu_id), torch.cuda.get_device_name())

    # Process dataset
    print("=====DATA PREPROCESSING=====")

    data = dataprep[dataset](split=split, pred_assign=psi)

    for i in range(num_tests):

        print("=====TEST {}=====".format(i + 1))
        print("=====TRAINING=====")

        if psi == 1:
            heckman_train_mse, heckman_test_mse, best_assignment = Heckman_FA(data_dic=data, c=c, max_epochs=max_epochs, lr=lr, B=B, device=device, rho_min=rho_min, rho_max=rho_max)
            naive_train_mse, naive_test_mse = naive_linear_regression(data_dic=data, assignment=best_assignment)

            print("Best assignment of prediction features: ", best_assignment)
        elif psi == 2:
            heckman_train_mse, heckman_test_mse, best_assignment = Heckman_FA(data_dic=data, c=c, max_epochs=max_epochs, lr=lr, B=B, device=device, rho_min=rho_min, rho_max=rho_max, star=True)
            naive_train_mse, naive_test_mse = naive_linear_regression(data_dic=data, assignment=best_assignment)

            print("Best assignment of prediction features: ", best_assignment)
        else:
            pred_feats = [0, 1, 2, 4, 7, 8, 10, 14, 15, 19, 20, 23, 24, 25]   # Replace with different set of prediction features to use
            naive_train_mse, naive_test_mse = naive_linear_regression(data_dic=data, assignment=pred_feats)
            heckman_train_mse, heckman_test_mse = train_eval_heckman(data_dic=data)

        print("Train MSE (naive linear regression): {:.6f}".format(naive_train_mse))
        print("Test MSE (naive linear regression): {:.6f}".format(naive_test_mse))
        print("Train MSE (Heckman's method): {:.6f}".format(heckman_train_mse))
        print("Test MSE (Heckman's method): {:.6f}".format(heckman_test_mse))

    print("=====TEST ENDED=====")
