function [ X, y, X_test, y_test ] = loadData()

X = importdata('assign2_dataset/page_blocks_train_feature.txt');
y = importdata('assign2_dataset/page_blocks_train_label.txt');
X_test = importdata('assign2_dataset/page_blocks_test_feature.txt');
y_test = importdata('assign2_dataset/page_blocks_test_label.txt');
end

