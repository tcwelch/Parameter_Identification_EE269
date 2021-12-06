## Overview
In our final project for EE 269, we performed second-order system parameter identification using three machine learning methods: K-Nearest Neighbors, Kernel Regression, and Neural Networks.

## Repo Structure
We generated our training, validation, and test data using ```data_generator.m```, storing training and validation data in ```cross_validation_fold1.csv, ... , cross_validation_fold5.csv```, while storing our test data in ```test_data.csv```.

We perform K-Nearest Neighbors estimation in ```knn_for_params.m``` and ```knn_sysID.m```; Kernel Regression in ```kernel_fitting.m``` and ```kernel_fitting.pdf```; and Neural Network prediction in ```MLP.ipynb``` and ```MLP.pdf```. We used ```cross_validation.m``` and ```test_rmse.m``` functions to get metrics for each set respectively.

## For further information

A great link for second-order systems (and an all-around solid website for any electronics topics): https://www.tutorialspoint.com/control_systems/control_systems_response_second_order.htm. 
This link helps explain cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html

You can also learn about [KNN](http://web.stanford.edu/class/ee269/Lecture4.pdf), [Kernel Regression](http://web.stanford.edu/class/ee269/Lecture11.pdf) and [Neural Networks](http://web.stanford.edu/class/ee269/Lecture15.pdf) on the EE 269 [course website](http://web.stanford.edu/class/ee269/index.html).
