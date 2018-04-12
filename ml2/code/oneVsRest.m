function [all_theta] = oneVsRest(X, y, num_labels, lambda, alpha, num_iters)

m = size(X, 1);
n = size(X, 2);


all_theta = zeros(num_labels, n);


%options = optimset('GradObj', 'on', 'MaxIter', 50, 'Display', 'off');
for c = 1:num_labels
    initial_theta = all_theta(c, :)';
    old_cost = costFunction(initial_theta, X, (y==c), lambda)
    %[all_theta(c,:)] = fminunc (@(t)(costFunction(t, X, (y == c), lambda)), initial_theta, options);
    [all_theta(c,:)] = gradientDescent(X, (y==c), initial_theta, lambda,  alpha, num_iters);
    %all_theta(c,:)'
    new_cost = costFunction(all_theta(c, :)', X, (y==c), lambda)
end
all_theta
end
