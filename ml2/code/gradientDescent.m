function [theta] = gradientDescent(X, y, theta, lambda, alpha, num_iters, class)

m = length(y); 
% costv = zeros(num_iters, 1);
for iter = 1:num_iters
    prediction = sigmoid(X * theta);
    theta = theta - alpha/m*X'*(prediction - y);
%     cost = costFunction(theta, X, y, lambda);
%     costv(iter) = cost;
end

% iterv= 1:iter;
% figure(class);
% plot(iterv, costv);
% xlabel('iterations');
% ylabel('cost');
end
