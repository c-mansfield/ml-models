data = readtable('iris.csv');

% Find 25% of the number of rows in data
nP = round(size(data,1) * 0.25);
rng(1)
% Shuffle data up
data_shuffled = data(randperm(size(data,1)), :);

% Split by 25% testing and 75% training
data_test = data_shuffled(1:1:nP, :);
data_train = data_shuffled(nP+1:1:end, :);
size(data_test)
size(data_train)

% Split examples and labels for both training and testing data
data_test_labels = categorical(data_test{:,'species'});
data_test_examples = data_test;
data_test_examples(:, 'species') = [];

data_train_labels = categorical(data_train{:,'species'});
data_train_examples = data_train;
data_train_examples(:, 'species') = [];

m = myknn.fit(data_train_examples, data_train_labels, 10);

% Predict class labels for test data with trained model
predictions = myknn.predict(m, data_test_examples);

% Generate confusion matrix to compare predictions to real class labels
[c,order] = confusionmat(predictions, data_test_labels);
c_chart = confusionchart(predictions, data_test_labels);

p = sum(diag(c)) / sum(c(1:1:end))
