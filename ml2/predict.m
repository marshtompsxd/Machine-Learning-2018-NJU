function p = predict(all_theta, X)

result = X * all_theta';
[~, p] = max(result, [], 2);

end
