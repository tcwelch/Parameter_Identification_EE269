%% Parameter Identification

% Authors: James, Rahul, Tom
% EE 269
% 11/9/2021

close all;
clear;
clc;

%% Creating dataset
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
X = zeros(num_samples,N);
Y = zeros(num_samples,2);
rand_cart2ind = reshape(randperm(num_samples),num_zeta,num_wn);
for i = 1:1:length(zeta)
    for j = 1:1:length(wn)
        sys = (wn(j)^2)/(s^2 + 2*zeta(i)*wn(j)*s + wn(j)^2);
        x = step(sys,t) + sigma*randn(N,1);
        y = [zeta(i),wn(j)];
        X(rand_cart2ind(i,j),:) = x';
        Y(rand_cart2ind(i,j),:) = y;
    end
end
%plot some examples to double check
figure(); hold on;
plot(t,X(1,:)); 
plot(t,X(2,:)); 
plot(t,X(3,:)); 
plot(t,X(4,:)); 
plot(t,X(5,:));
plot(t,X(6,:));
plot(t,X(7,:)); 
title('Step Response');
xlabel('Time (sec)');
ylabel('H(t)');

%slicing data into validation sets
slice_size = 200;
X1 = X(1:slice_size,:);
X2 = X(slice_size+1:2*slice_size,:);
X3 = X(2*slice_size+1:3*slice_size,:);
X4 = X(3*slice_size+1:4*slice_size,:);
X5 = X(4*slice_size+1:end,:);

Y1 = Y(1:slice_size,:);
Y2 = Y(slice_size+1:2*slice_size,:);
Y3 = Y(2*slice_size+1:3*slice_size,:);
Y4 = Y(3*slice_size+1:4*slice_size,:);
Y5 = Y(4*slice_size+1:end,:);

%saving data
save('step_response_data.mat','X1','X2','X3','X4','X5','Y1','Y2','Y3','Y4','Y5');

zeta1 = Y1(:,1);
wn1 = Y1(:,2);
T1 = table(zeta1,wn1,X1);
writetable(T1,'step_response_data1.csv');

zeta2 = Y2(:,1);
wn2 = Y2(:,2);
T2 = table(zeta2,wn2,X2);
writetable(T2,'step_response_data2.csv');

zeta3 = Y3(:,1);
wn3 = Y3(:,2);
T3 = table(zeta3,wn3,X3);
writetable(T3,'step_response_data3.csv');

zeta4 = Y4(:,1);
wn4 = Y4(:,2);
T4 = table(zeta4,wn4,X4);
writetable(T4,'step_response_data4.csv');

zeta5 = Y5(:,1);
wn5 = Y5(:,2);
T5 = table(zeta5,wn5,X5);
writetable(T5,'step_response_data5.csv');

