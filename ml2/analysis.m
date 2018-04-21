function [ precision, recall ] = analysis( y_test, pred, num_labels )

precision = zeros(num_labels, 1);
recall = zeros(num_labels, 1);

for i = 1 : num_labels
    TP_plus_FN_v = (y_test == i);
    TP_plus_FP_v = (pred == i);
    TP_v = TP_plus_FN_v .* TP_plus_FP_v;
    precision(i) = sum(TP_v)/sum(TP_plus_FP_v);
    recall(i) = sum(TP_v)/sum(TP_plus_FN_v);
    fprintf('Test Set Precision of class %d: %f%%\n',i, precision(i)*100);
    fprintf('Test Set Recall of class %d: %f%%\n',i, recall(i)*100);
end

fprintf('Test Set Accuracy: %f%%\n', mean(double(pred == y_test)) * 100);
end

