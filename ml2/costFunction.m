function J= costFunction( theta, X, y )

m = length(y);
prediction = sigmoid(X * theta);
J = 1/m * sum( - y .* log(prediction) - (1-y) .* log(1-prediction) );

end

