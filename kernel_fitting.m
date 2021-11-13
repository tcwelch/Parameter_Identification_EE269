%% Parameter Identification

% Authors: James, Rahul, Tom
% EE 269
% 11/9/2021

close all;
clear;
clc;
% List of kernels to try:http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#laplacian
%% Fitting Linear Function
%% Finding Lambda
load("cross_validation_data.mat");
kernel = @linear_kernel;
lambdas = logspace(-2,5,8);
fig_handle1 = find_lambda(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambdas,[]);
%% Running on Test Set
load("test_data.mat");
load("cross_validation_data.mat");
kernel = @linear_kernel;
lambda = 10;
prmse = test_evaluation(X_test,Y_test,X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
        Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,[]); 
disp(['Percent RMS Error (on Test Set with Linear Fit) = ' num2str(prmse) '%']);

%% Fitting Polynomial Function
%% Finding Lambda
load("cross_validation_data.mat");
kernel = @polynomial_kernel;
lambdas = logspace(3,10,8);
order = 4;
fig_handle1 = find_lambda(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambdas,order);
%% Finding polynomial degree order
load("cross_validation_data.mat");
kernel = @polynomial_kernel;
lambda = 10^9;
orders = 2:6;
fig_handle2 = find_hyperparam(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,orders,'Polynomial Order');

%% Running on Test Set
load("test_data.mat");
load("cross_validation_data.mat");
kernel = @polynomial_kernel;
lambda = 10^9;
order = 4;
prmse = test_evaluation(X_test,Y_test,X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
        Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,order);
disp(['Percent RMS Error (on Test Set with Polynomial Kernel) = ' num2str(prmse) '%']);

%% Fitting Gassian Kernel
%% Finding Lambda
load("cross_validation_data.mat");
kernel = @gaussian_kernel;
lambdas = logspace(-3,4,8);
sigma = 0.1;
fig_handle1 = find_lambda(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambdas,sigma);
%% Finding Sigma
load("cross_validation_data.mat");
kernel = @gaussian_kernel;
lambda = 10^-3;
sigmas = logspace(-3,4,8);
fig_handle2 = find_hyperparam(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,sigmas,'Gaussian Sigma Variables');

%% Running on Test Set
load("test_data.mat");
load("cross_validation_data.mat");
kernel = @gaussian_kernel;
lambda = 10^-3;
sigma = 2;
prmse = test_evaluation(X_test,Y_test,X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
        Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,sigma);
disp(['Percent RMS Error (on Test Set with Gaussian Kernel) = ' num2str(prmse) '%']);

%% Fitting Laplace Kernel
%% Finding Lambda
load("cross_validation_data.mat");
kernel = @laplace_kernel;
lambdas = logspace(-3,4,8);
sigma = 0.1;
fig_handle1 = find_lambda(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambdas,sigma);
%% Finding Sigma
load("cross_validation_data.mat");
kernel = @laplace_kernel;
lambda = 10^-1;
sigmas = logspace(-3,4,8);
fig_handle2 = find_hyperparam(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,sigmas,'Gaussian Sigma Variable');

%% Running on Test Set
load("test_data.mat");
load("cross_validation_data.mat");
kernel = @laplace_kernel;
lambda = 10^-1;
sigma = 10;
prmse = test_evaluation(X_test,Y_test,X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
        Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,sigma);
disp(['Percent RMS Error (on Test Set with Laplace Kernel) = ' num2str(prmse) '%']);

%% Fitting Histogram Intersection Kernel
%% Finding Lambda
load("cross_validation_data.mat");
kernel = @histogram_intersection_kernel;
lambdas = logspace(-2,5,8);
fig_handle1 = find_lambda(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambdas,[]);
%% Running on Test Set
load("test_data.mat");
load("cross_validation_data.mat");
kernel = @histogram_intersection_kernel;
lambda = 1;
prmse = test_evaluation(X_test,Y_test,X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
        Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,[]); 
disp(['Percent RMS Error (on Test Set with Histogram Intersection Kernel) = ' num2str(prmse) '%']);
%% Fitting Wavelet Kernel
%% Finding Lambda
load("cross_validation_data.mat");
kernel = @wavelet_kernel;
lambdas = logspace(-2,5,8);
fig_handle1 = find_lambda(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambdas,[]);
%% Running on Test Set
load("test_data.mat");
load("cross_validation_data.mat");
kernel = @wavelet_kernel;
lambda = 1;
prmse = test_evaluation(X_test,Y_test,X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
        Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,[]); 
disp(['Percent RMS Error (on Test Set with Wavelet Kernel) = ' num2str(prmse) '%']);

%% Functions
function k_xi_xj = linear_kernel(xi,xj,hyperparam)
    k_xi_xj = reshape(xi,1,[])*reshape(xj,[],1);
end

function k_xi_xj = polynomial_kernel(xi,xj,order)
    k_xi_xj = (1 + reshape(xi,1,[])*reshape(xj,[],1))^order;
end


function k_xi_xj = gaussian_kernel(xi,xj,sigma)
    k_xi_xj = exp(-(norm(reshape(xi,1,[]) - reshape(xj,1,[]),2)^2 )/(2*sigma^2));
end

function k_xi_xj = laplace_kernel(xi,xj,sigma)
    k_xi_xj = exp(-norm(reshape(xi,1,[]) - reshape(xj,1,[]),2)/sigma);
end

function k_xi_xj = histogram_intersection_kernel(xi,xj,hyperparam)
    k_xi_xj = sum(min(reshape(xi,[],1),reshape(xj,[],1)));
end

function k_xi_xj = wavelet_kernel(xi,xj,hyperparam)
     k_xi_xj = prod(wavelet(reshape(xi,1,[])).*wavelet(reshape(xi,1,[])));
end

function y = wavelet(x)
    y = cos(1.75*x).*exp(-(x.^2)/2);
end

%rows of X are samples
function K = form_K(X_train,X_test,kernel,hyper_param)
    N = length(X_train(:,1));
    M = length(X_test(:,1));
    K = zeros(M,N);
    for i = 1:1:M
        for j = 1:1:N
            K(i,j) = kernel(X_test(i,:),X_train(j,:),hyper_param);
        end
    end
end

function y_test_pred = predict(X_train,Y_train,X_test,lambda, kernel,hyper_param)
    K_train = form_K(X_train,X_train,kernel,hyper_param);
    alpha = inv((1/lambda)*K_train + eye(size(K_train)))*Y_train;
    K_test = form_K(X_train,X_test,kernel,hyper_param);
    y_test_pred = (1/lambda)*K_test*alpha;
end

function avg_prmse = cross_validate(X1,X2,X3,X4,X5,Y1,Y2,Y3,Y4,Y5,kernel,lambda,hyperparam)
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
        Y_test_pred = predict(X_train,Y_train,X_test,lambda, kernel,hyperparam);
        prmse = prmse + calc_percent_rmse(Y_test,Y_test_pred);
    end
    avg_prmse = prmse/num_folds;
end

function prmse = test_evaluation(X_test,Y_test,X1,X2,X3,X4,X5,Y1,Y2,Y3,Y4,Y5,kernel,lambda,hyperparam)
    X_train = [X1;X2;X3;X4;X5];
    Y_train = [Y1;Y2;Y3;Y4;Y5];
    Y_test_pred = predict(X_train,Y_train,X_test,lambda, kernel,hyperparam);
    prmse = calc_percent_rmse(Y_test,Y_test_pred);
end

function fig_handle = find_lambda(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambdas,hyperparam)
    avg_prmses = zeros(length(lambdas),1);
    figure();
    for i = 1:1:length(lambdas)
        lambda = lambdas(i);
        avg_prmses(i) = cross_validate(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,hyperparam);
    end
    semilogx(lambdas,avg_prmses,'*');
    title('5-Fold Cross Validation Percent RMS Error vs. Lambda');
    xlabel('Lambda'); ylabel('Average Percent RMSE (%)');
    fig_handle = gcf;
end

function fig_handle = find_hyperparam(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,hyperparams,hyperparam_name)
    avg_prmses = zeros(length(hyperparams),1);
    figure();
    for i = 1:1:length(hyperparams)
        hyperparam = hyperparams(i);
        avg_prmses(i) = cross_validate(X_valid1,X_valid2,X_valid3,X_valid4,X_valid5,...
            Y_valid1,Y_valid2,Y_valid3,Y_valid4,Y_valid5,kernel,lambda,hyperparam);
    end
    plot(hyperparams,avg_prmses,'*');
    title(['5-Fold Cross Validation Percent RMS Error vs. ' hyperparam_name]);
    xlabel(hyperparam_name); ylabel('Average Percent RMSE (%)');
    fig_handle = gcf;
end

function prmse = calc_percent_rmse(x_actual,x_corrupted)
    N = length(x_actual);
    prmse = sqrt((1/N)*sum((reshape(x_corrupted,[],1) - reshape(x_actual,[],1)).^2))*(100*N/sum(reshape(x_actual,[],1)));
end
