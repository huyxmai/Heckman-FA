Conduct the whole experiment by running the 'experiment_tabular.py' file. In particular, run Heckman-FA by executing the command
python3 -W ignore experiment_tabular.py --dataset [DATASET_NAME] --prediction_assignment 1 --c [c] --T [NUMBER_OF_EPOCHS] --alpha [LEARNING_RATE] --B [NUMBER OF GUMBEL-SOFTMAX SAMPLES] --rho_range [RHO_MIN] [RHO_MAX]

To run Heckman-FA*, change the --prediction-assignment argument to 2.

To test a specific set of prediction features (using '--prediction_assignment 0' as the argument), enter the corresponding indices of the features as a list titled 'pred_feats'. An example is shown in the 'data_preprocessing_tabular.py' file in the 'preprocess_crime' function.