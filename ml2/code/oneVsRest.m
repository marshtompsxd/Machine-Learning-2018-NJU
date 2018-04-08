function [all_theta] = oneVsRest(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);


all_theta = zeros(num_labels, n);


options = optimset('GradObj', 'on', 'MaxIter', 50);
for c = 1:num_labels
    initial_theta = all_theta(c, :)';
    %oldJ = costFunction(all_theta(c, :)', X, (y==c), lambda)
    [all_theta(c,:)] = fminunc (@(t)(costFunction(t, X, (y == c), lambda)), initial_theta, options);
    %newJ = costFunction(all_theta(c, :)', X, (y==c), lambda)
end

end
