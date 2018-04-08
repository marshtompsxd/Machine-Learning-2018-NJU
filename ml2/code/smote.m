function [ X_new, y_new ] = smote( X, y, k, num_labels )
%SMOTE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
category_size = zeros(1, num_labels);

for i=1 : num_labels
    for j = 1 : size(y)
        if(y(j) == i)
            category_size(i) = category_size(i) + 1;
        end
    end
end

max_num = max(category_size);
X_new = [];
y_new = [];
for i=1 : num_labels
    Xs = X(y==i, : );
    N = fix(max_num/category_size(i));
    Xs_new = smoteAux(Xs, N - 1, k);
    ys_new = ones(category_size(i)*N, 1)*i;
    X_new = [X_new; Xs_new];
    y_new = [y_new; ys_new];
end
end

