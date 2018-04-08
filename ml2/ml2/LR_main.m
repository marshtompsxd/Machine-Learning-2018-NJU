
clear ; close all; clc
load('data.mat');
num_labels = 5;
k = 5;
[X, y] = smote(X, y, 5, num_labels);
m = size(X, 1);

lambda = 0.1;
[X, mu, sigma] = normalize(X);
X = [ones(m, 1) X];

[all_theta] = oneVsRest(X, y, num_labels, lambda);
%all_theta

pred = predict(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

X_test = bsxfun(@minus, X_test, mu);
X_test = bsxfun(@rdivide, X_test, sigma);
X_test = [ones(size(X_test, 1), 1), X_test]; 

pred = predict(all_theta, X_test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);