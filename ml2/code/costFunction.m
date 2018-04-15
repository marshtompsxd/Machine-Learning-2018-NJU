function J= costFunction( theta, X, y, lambda )

m = length(y);
% grad = zeros(size(theta));
prediction = sigmoid(X * theta);
J = 1/m * sum( - y .* log(prediction) - (1-y) .* log(1-prediction) );
  % + lambda/(2*m) * sum(theta(2:end).^2);
% for j = 1 : size(theta)
%         if j == 1
%             grad(j) = 1 / m *  (prediction - y)' * X(:, j);
%         else
%             grad(j) = 1 / m * (prediction - y)' * X(:, j) + lambda / m * theta(j);
%         end
% end

end

