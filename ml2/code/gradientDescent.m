function [theta] = gradientDescent(X, y, theta, lambda, alpha, num_iters)

m = length(y); 

for iter = 1:num_iters
    new_theta = theta;
    prediction = sigmoid(X * theta);
    new_theta(1) = theta(1) - alpha/m*(prediction - y)' * X(:, 1);
    for i = 2 : size(X, 2)
        new_theta(i) = theta(i) - alpha/m*(prediction - y)' * X(:, i) - alpha*lambda/m*new_theta(i);
    end
    theta = new_theta;
    %cost = costFunction(theta, X, y, lambda)
end

end
