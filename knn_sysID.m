%% Parameter Identification with KNN

% Authors: James, Rahul, Tom
% EE 269
% 11/9/2021

%% Load data
close all; clear all;
load('cross_validation_data.mat');

%% 1. No feature extraction - uniform average: Training
use_uniform = 1; % if 1, use uniform average for knn regression, if 0 use weighted average
k_max = 25; % largest number of surrounding neighbors to compute average with
k_values = 1:1:k_max;
prsme_values_1 = zeros(length(k_values),1);

for k = k_values
   prsme_values_1(k) = cross_validate(X_valid1, X_valid2, X_valid3, X_valid4, X_valid5, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, k, use_uniform); 
end

% Identify best k value
[~, best_k1] = min(prsme_values_1);
%% 2. No feature extraction - uniform average: Testing
load('test_data.mat');
prsme_value_2 = test_evaluation(X_test,Y_test,X_valid1, X_valid2, X_valid3, X_valid4, X_valid5, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, best_k1, use_uniform);
%% 3. No feature extraction - weighted average: Training
use_uniform = 0; % if 1, use uniform average for knn regression, if 0 use weighted average
k_max = 25; % largest number of surrounding neighbors to compute average with
k_values = 1:1:k_max;
prsme_values_3 = zeros(length(k_values),1);

for k = k_values
   prsme_values_3(k) = cross_validate(X_valid1, X_valid2, X_valid3, X_valid4, X_valid5, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, k, use_uniform); 
end

% Identify best k value
[~, best_k2] = min(prsme_values_3);
%% 4. No feature extraction - weighted average: Testing
prsme_value_4 = test_evaluation(X_test,Y_test,X_valid1, X_valid2, X_valid3, X_valid4, X_valid5, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, best_k2, use_uniform);
%% 5. FFT - uniform average: Training
use_uniform = 1; % if 1, use uniform average for knn regression, if 0 use weighted average
k_max = 25; % largest number of surrounding neighbors to compute average with
k_values = 1:1:k_max;
prsme_values_1 = zeros(length(k_values),1);

% Take FFT 
X_valid1_fft = abs(fft(X_valid1,[],2));
X_valid2_fft = abs(fft(X_valid2,[],2));
X_valid3_fft = abs(fft(X_valid3,[],2));
X_valid4_fft = abs(fft(X_valid4,[],2));
X_valid5_fft = abs(fft(X_valid5,[],2));

for k = k_values
   prsme_values_1(k) = cross_validate(X_valid1_fft, X_valid2_fft, X_valid3_fft, X_valid4_fft, X_valid5_fft, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, k, use_uniform); 
end

% Identify best k value
[~, best_k1] = min(prsme_values_1);
%% 6. FFT - uniform average: Testing
load('test_data.mat');
X_test_fft = abs(fft(X_test,[],2));
prsme_value_2 = test_evaluation(X_test_fft,Y_test,X_valid1_fft, X_valid2_fft, X_valid3_fft, X_valid4_fft, X_valid5_fft, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, k, use_uniform); 
%% 7. FFT - weighted average: Training
use_uniform = 0; % if 1, use uniform average for knn regression, if 0 use weighted average
k_max = 25; % largest number of surrounding neighbors to compute average with
k_values = 1:1:k_max;
prsme_values_3 = zeros(length(k_values),1);

for k = k_values
   prsme_values_3(k) = cross_validate(X_valid1_fft, X_valid2_fft, X_valid3_fft, X_valid4_fft, X_valid5_fft, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, k, use_uniform); 
end

[~, best_k2] = min(prsme_values_3);
%% 8. FFT - weighted average: Testing
prsme_value_4 = test_evaluation(X_test_fft,Y_test,X_valid1_fft, X_valid2_fft, X_valid3_fft, X_valid4_fft, X_valid5_fft, ...
    Y_valid1, Y_valid2, Y_valid3, Y_valid4, Y_valid5, k, use_uniform); 
%% Functions
function prmse = calc_percent_rmse(x_actual,x_corrupted)
    N = length(x_actual);
    prmse = sqrt((1/N)*sum((reshape(x_corrupted,[],1) - reshape(x_actual,[],1)).^2))*(100*N/sum(reshape(x_actual,[],1)));
end

function r_sqd_value = r_sqd(x_actual, x_corrupted)
    x_actual = reshape(x_actual,[],1);
    x_corrupted = reshape(x_corrupted,[],1);
    e = x_actual - x_corrupted;
    x_actual_bar = mean(x_actual);
    SSres = sum(e.^2);
    SStot = sum((x_actual - x_actual_bar).^2);
    r_sqd_value = 1 - (SSres/SStot);
end

function avg_prmse_knn = cross_validate(X1,X2,X3,X4,X5,Y1,Y2,Y3,Y4,Y5,k,use_uniform)
    X = {X1,X2,X3,X4,X5};
    Y = {Y1,Y2,Y3,Y4,Y5};
    num_folds = 5;
    folds = 1:1:num_folds;
    prmse = 0;
    for i = folds
        idxs = folds;
        idxs(idxs == i) = [];
        [x1,x2,x3,x4] = X{idxs};
        [y1,y2,y3,y4] = Y{idxs};
        X_train = [x1;x2;x3;x4];
        X_test = X{i};
        Y_train = [y1;y2;y3;y4];
        Y_test = Y{i};
        % Run prediction algorithm
        Y_test_pred = knn_for_params(X_train,X_test, k, Y_train,use_uniform);
        prmse = prmse + calc_percent_rmse(Y_test,Y_test_pred);
    end
    avg_prmse_knn = prmse/num_folds;
end

function prmse_knn = test_evaluation(X_test,Y_test,X1,X2,X3,X4,X5,Y1,Y2,Y3,Y4,Y5,best_k,use_uniform)
    X_train = [X1;X2;X3;X4;X5];
    Y_train = [Y1;Y2;Y3;Y4;Y5];
    Y_test_pred = knn_for_params(X_train,X_test, best_k, Y_train,use_uniform);
    prmse_knn = calc_percent_rmse(Y_test,Y_test_pred);
end