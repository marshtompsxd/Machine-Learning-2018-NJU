function p = predict(all_theta, X)


m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);
       

result = X * all_theta';
[temp p] = max(result, [], 2);


end
