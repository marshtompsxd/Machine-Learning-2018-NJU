fprintf('clearing memory space...\n');
clear ; close all; clc

num_labels = 5;
k = 10;
lambda = 0.1;
alpha = 5;
num_iters = 1000;

fprintf('loading data...\n');
[X, y, X_test, y_test] = loadData();

fprintf('oversampling...\n');
[X, y] = smote(X, y, 5, num_labels);
m = size(X, 1);

fprintf('normalizing the data...\n');
[X, mu, sigma] = normalize(X);
X = [ones(m, 1) X];


fprintf('training...\n');
[all_theta] = oneVsRest(X, y, num_labels, alpha, num_iters);

fprintf('predicting for training set...\n');
pred = predict(all_theta, X);

fprintf('Training Set Accuracy: %f%%\n', mean(double(pred == y)) * 100);

X_test = bsxfun(@minus, X_test, mu);
X_test = bsxfun(@rdivide, X_test, sigma);
X_test = [ones(size(X_test, 1), 1), X_test]; 

fprintf('predicting for test set...\n');
pred = predict(all_theta, X_test);
analysis(y_test, pred, num_labels);
