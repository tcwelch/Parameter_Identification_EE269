%% Parameter Identification

% Authors: James, Rahul, Tom
% EE 269
% 11/9/2021

close all;
clear;
clc;

%% Creating Validation Dataset
s = tf('s');
fs = 100;
N = 500; %Length of Signal
t = 0:(1/fs):(N-1)*(1/fs);
wn = (2*pi)*(1:1:50);
zeta = linspace(0,1,20);
num_wn = length(wn);
num_zeta = length(zeta);
num_samples = num_wn*num_zeta;
sigma = 0.01;
X_valid = zeros(num_samples,N);
Y_valid = zeros(num_samples,2);
rand_cart2ind = reshape(randperm(num_samples),num_zeta,num_wn);
for i = 1:1:length(zeta)
    for j = 1:1:length(wn)
        sys = (wn(j)^2)/(s^2 + 2*zeta(i)*wn(j)*s + wn(j)^2);
        x = step(sys,t) + sigma*randn(N,1);
        y = [zeta(i),wn(j)];
        X_valid(rand_cart2ind(i,j),:) = x';
        Y_valid(rand_cart2ind(i,j),:) = y;
    end
end
%plot some examples to double check
figure(); hold on;
plot(t,X_valid(1,:)); 
plot(t,X_valid(2,:)); 
plot(t,X_valid(3,:)); 
plot(t,X_valid(4,:)); 
plot(t,X_valid(5,:));
plot(t,X_valid(6,:));
plot(t,X_valid(7,:)); 
title('Step Response');
xlabel('Time (sec)');
ylabel('H(t)');

%slicing data into validation sets
slice_size = 200;
X_valid1 = X_valid(1:slice_size,:);
X_valid2 = X_valid(slice_size+1:2*slice_size,:);
X_valid3 = X_valid(2*slice_size+1:3*slice_size,:);
X_valid4 = X_valid(3*slice_size+1:4*slice_size,:);
X_valid5 = X_valid(4*slice_size+1:end,:);

Y_valid1 = Y_valid(1:slice_size,:);
Y_valid2 = Y_valid(slice_size+1:2*slice_size,:);
Y_valid3 = Y_valid(2*slice_size+1:3*slice_size,:);
Y_valid4 = Y_valid(3*slice_size+1:4*slice_size,:);
Y_valid5 = Y_valid(4*slice_size+1:end,:);

%saving data
save(['cross_validation_data.mat'],'X_valid1','X_valid2','X_valid3','X_valid4','X_valid5','Y_valid1','Y_valid2','Y_valid3','Y_valid4','Y_valid5');

zeta1 = Y_valid1(:,1);
wn1 = Y_valid1(:,2);
T1 = table(zeta1,wn1,X_valid1);
writetable(T1,'cross_validation_fold1.csv');

zeta2 = Y_valid2(:,1);
wn2 = Y_valid2(:,2);
T2 = table(zeta2,wn2,X_valid2);
writetable(T2,'cross_validation_fold2.csv');

zeta3 = Y_valid3(:,1);
wn3 = Y_valid3(:,2);
T3 = table(zeta3,wn3,X_valid3);
writetable(T3,'cross_validation_fold3.csv');

zeta4 = Y_valid4(:,1);
wn4 = Y_valid4(:,2);
T4 = table(zeta4,wn4,X_valid4);
writetable(T4,'cross_validation_fold4.csv');

zeta5 = Y_valid5(:,1);
wn5 = Y_valid5(:,2);
T5 = table(zeta5,wn5,X_valid5);
writetable(T5,'cross_validation_fold5.csv');

%% Creating Test Dataset
s = tf('s');
fs = 100;
N = 500; %Length of Signal
t = 0:(1/fs):(N-1)*(1/fs);
wn = (2*pi)*(1+49*rand([20,1]));
zeta = rand([10,1]);
num_wn = length(wn);
num_zeta = length(zeta);
num_samples = num_wn*num_zeta;
sigma = 0.01;
X_test = zeros(num_samples,N);
Y_test = zeros(num_samples,2);
rand_cart2ind = reshape(randperm(num_samples),num_zeta,num_wn);
for i = 1:1:length(zeta)
    for j = 1:1:length(wn)
        sys = (wn(j)^2)/(s^2 + 2*zeta(i)*wn(j)*s + wn(j)^2);
        x = step(sys,t) + sigma*randn(N,1);
        y = [zeta(i),wn(j)];
        X_test(rand_cart2ind(i,j),:) = x';
        Y_test(rand_cart2ind(i,j),:) = y;
    end
end
%plot some examples to double check
figure(); hold on;
plot(t,X_test(1,:)); 
plot(t,X_test(2,:)); 
plot(t,X_test(3,:)); 
plot(t,X_test(4,:)); 
plot(t,X_test(5,:));
plot(t,X_test(6,:));
plot(t,X_test(7,:)); 
title('Step Response');
xlabel('Time (sec)');
ylabel('H(t)');

%saving data
save('test_data.mat','X_test','Y_test');

zeta = Y_test(:,1);
wn = Y_test(:,2);
T = table(zeta,wn,X_test);
writetable(T,'test_data.csv');

