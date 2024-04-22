# On Prediction Feature Assignment in the Heckman Selection Model

This is the source code of Heckman-FA, which was proposed in "On Prediction Feature Assignment in the Heckman Selection Model".

Under missing-not-at-random (MNAR) sample selection bias, the performance of a prediction model is often degraded. This paper focuses on one classic instance of MNAR sample selection bias where a subset of samples have non-randomly missing outcomes. The Heckman selection model and its variants have commonly been used to handle this type of sample selection bias. The Heckman model uses two separate equations to model the prediction and selection of samples, where the selection features include all prediction features. When using the Heckman model, the prediction features must be properly chosen from the set of selection features. However, choosing the proper prediction features is a challenging task for the Heckman model. This is especially the case when the number of selection features is large. Existing approaches that use the Heckman model often provide a manually chosen set of prediction features. In this paper, we propose Heckman-FA as a novel data-driven framework for obtaining prediction features for the Heckman model. Heckman-FA first trains an assignment function that determines whether or not a selection feature is assigned as a prediction feature. Using the parameters of the trained function, the framework extracts a suitable set of prediction features based on the goodness-of-fit of the prediction model given the chosen prediction features and the correlation between noise terms of the prediction and selection equations. Experimental results on real-world datasets show that Heckman-FA produces a robust regression model under MNAR sample selection bias.

## Usage

### Environment

1. Clone repository
```
git clone https://github.com/huyxmai/Heckman-FA.git
cd Heckman-FA
```
2. Create new environment
```
conda create -n heckman-fa python=3.9.2
```
3. Activate environment
```
conda activate heckman-fa
```
4. Install packages
```
pip install -r requirements.txt
```

### Heckman-FA

Run Heckman-FA by executing the command
```
python3 -W ignore experiment_tabular.py --dataset [DATASET_NAME]\ 
--prediction_assignment 1\ 
--c [c] --T [NUMBER_OF_EPOCHS]\ 
--alpha [LEARNING_RATE]\ 
--B [NUMBER OF GUMBEL-SOFTMAX SAMPLES]\ 
--rho_range [RHO_MIN] [RHO_MAX]
```

To run Heckman-FA*, change the --prediction-assignment argument to 2.

To test a specific set of prediction features (using '--prediction_assignment 0' as the argument), enter the corresponding indices of the features as a list titled 'pred_feats'. An example is shown in the 'data_preprocessing_tabular.py' file in the 'preprocess_crime' function.
