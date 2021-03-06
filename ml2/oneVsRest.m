function [all_theta] = oneVsRest(X, y, num_labels, alpha, num_iters)

%m = size(X, 1);
n = size(X, 2);


all_theta = zeros(num_labels, n);
% all_theta2 = zeros(num_labels, n);

%options = optimset('GradObj', 'on', 'MaxIter', 50, 'Display', 'off');
for c = 1:num_labels
    fprintf('Training model for class %d...\n', c);
    initial_theta = all_theta(c, :)';
%     initial_theta2 = all_theta2(c, :)';
%     old_cost = costFunction(initial_theta, X, (y==c), lambda)
%     [all_theta2(c,:)] = fminunc (@(t)(costFunction(t, X, (y == c), lambda)), initial_theta2, options);
    [all_theta(c,:)] = gradientDescent(X, (y==c), initial_theta, alpha, num_iters, c);
%     all_theta(c,:)'
%     new_cost = costFunction(all_theta(c, :)', X, (y==c), lambda)
%     new_cost2 = costFunction(all_theta2(c, :)', X, (y==c), lambda)
end
% all_theta
% all_theta2
% all_theta - all_theta2
end
