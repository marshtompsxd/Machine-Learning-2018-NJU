function [ X, y, X_test, y_test ] = loadData()

X = load('./assign2_dataset/page_blocks_train_feature.txt');
y = load('./assign2_dataset/page_blocks_train_label.txt');
X_test = load('./assign2_dataset/page_blocks_test_feature.txt');
y_test = load('./assign2_dataset/page_blocks_test_label.txt');
end

