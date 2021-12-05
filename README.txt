## Overview
In this project we performed second-order system parameter identification using three machine learning methods: K-Nearest Neighbors, Kernel Regression, and Neural Networks.

## Repo Structure
We generated our training, validation, and test data using ```data_generator.m```, storing training and validation data in ```cross_validation_fold1.csv, ... , cross_validation_fold5.csv```, while storing our test data in ```test_data.csv```.

We perform K-Nearest Neighbors estimation in ```knn_for_params.m``` and ```knn_sysID.m```; Kernel Regression in ```kernel_fitting.m``` and ```kernel_fitting.pdf```; and Neural Network prediction in ```MLP.ipynb``` and ```MLP.pdf```.

- The data is available as .mat or .csv (the data is the same between the .csv files or the .mat files are used)
- It has been shuffled and pre-spliced into 5 seperate cross validation
	- These 5 slices are in seperate files for the .csv format and are in
	one file  with slices of corresponding X,Y sets denoted with 1-5 like this: X1,Y1
- I changed the noise to have sigma = 0.01 rather than 0.1 because 0.1 proved to be too large to distinguish anything
- Do not generate new data (i.e. do NOT run generate_data) since we want the dataset to stay constant
- The cross validation folds are used for tuning hyperparameters and the test set is used for final results
  of the algorithm trained on all 5 folds of the validation set combined.
        - Use cross_validation.m and test_rmse.m functions to get metrics for each set respectively
        - This link helped explain cross validation...https://scikit-learn.org/stable/modules/cross_validation.html
