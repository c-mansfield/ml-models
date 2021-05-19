% Object to train and predict data label values using Decision Trees model

classdef mytree
    methods(Static)

        %% Responsible for processes that train data set for Decision tree model
        % Parameters: data-set examples without class labels, data-set class labels
        % Returns: static data object m with all trained Decision tree attributes
        function m = fit(train_examples, train_labels)

            % Static reusable data structure to represent individual nodes in a
            % tree, will reuse it for each node
			emptyNode.number = [];    % Node unique number(ID)
            emptyNode.examples = [];  % Associated training examples
            emptyNode.labels = [];    % Associated training class labels
            emptyNode.prediction = [];  % Predictions based on any class label the node holds
            emptyNode.impurityMeasure = [];   % Measure of impurity of any class labels held by a node, used when deciding whether to split
            emptyNode.children = {};    % If split, stores two child nodes, dividing its training examples
            emptyNode.splitFeature = [];  % Feature with name and value that describes the split
            emptyNode.splitFeatureName = [];
            emptyNode.splitValue = [];

            % Empty individual node data structure
            m.emptyNode = emptyNode;

            % Creates a root node(Very top mode in the tree) using node
            % data structure created above
            r = emptyNode;
            r.number = 1;       % Sets node number to 1 because root node
            r.labels = train_labels;   % Copys all training class labels over, due to being root node all associative
            r.examples = train_examples;  % Copys all training examples over, due to being root node all associative
            r.prediction = mode(r.labels);  % Generates single class label prediction, used if cannot split tree/data

            % Set additional model information
            m.min_parent_size = 10;         % Sets minimum number of examples a node must have before we consider splitting
            m.unique_classes = unique(r.labels);   % The class labels we are dealing with, eg. Benign, Malignant
            m.feature_names = train_examples.Properties.VariableNames;   % All feature/column titles
			m.nodes = 1;        % Current number of nodes in tree, starts at 1 because at root node will increment for each node
            m.N = size(train_examples,1);   % Size of the training data

            % Genarates the decision tree for the data, finding each node
            % recursively
            m.tree = mytree.trySplit(m, r);

        end

        %% Recursive function to generate the tree, each iteration tests whether a node can be split, with a reduced overall impurity with its other class labels
        % Parameters: m model data structure with model attributes, current
        % node(Start is root node created in 'fit')
        % Returns: Newly created node
        function node = trySplit(m, node)

            % Checks whether current node is large enough to be a candidate
            % for splitting and become a parent, minimum requirment set to 10 by default
            if size(node.examples, 1) < m.min_parent_size
				return
			end

            % Finds current impurity within class labels, it is the purity
            % of the class labels that gives us cofidence in our
            % predictions, 0 is a pure node.
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);

            % Iterates over number of features
            % Looking at possible ways we can split the training data for
            % the node, conidering splitting on every feature
            for i=1:size(node.examples,2)

				%fprintf('evaluating possible splits on feature %d/%d\n', i, size(node.examples,2));

                % Sort examples based on current feature from low to high
                % Use the indices to get the class labels for each of the
                % feature rows
				[ps,n] = sortrows(node.examples,i);
                ls = node.labels(n);

                % Create biggest reduction array to keep track of the feature example with the biggest
                % reduction in impurity, recording the reduction value and the index of the feature value
                biggest_reduction(i) = -Inf;
                biggest_reduction_index(i) = -1;
                biggest_reduction_value(i) = NaN;

                % Iterate through size of feature examples - 1, if all values
                % are unique then the number of possible splits is always 1
                % less than number of examples
                for j=1:(size(ps,1)-1)

                    % Checks if next feature example is the same as current
                    % one, if same value isn't unique so skips over example
                    if ps{j,i} == ps{j+1,i}
                        continue;
                    end

                    % Find the reduction impurity, calculates the GDI for the two collections of class
                    % labels created by a given split, adds them together, and subtracts them from
                    % the GDI score for the original table, if result is positive then the split
                    % produces a reduction in impurity
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end)));

                    % Checks if split will reduce the overall impurity, split will be based on
                    % feature value that brought greatest reduction in impurity,
                    % checking each reduction by the overall biggest reduction
                    if this_reduction > biggest_reduction(i)
                        biggest_reduction(i) = this_reduction;
                        biggest_reduction_index(i) = j;
                    end
                end

            end

            % Finds the feature value and its index of the element with the
            % largest impurity reduction
            [winning_reduction,winning_feature] = max(biggest_reduction);
            winning_index = biggest_reduction_index(winning_feature);

            % Finds if is possible a reduction in impurity through
            % splitting by checking if the winning reduction is greater than 0
            % Proceeds to split nodes into two further ones if greater
            if winning_reduction <= 0
                % Leaves the current node alone when winning reduction less
                % than 0
                return
            else

                % Sort rows by winning feature values as currently sorted
                % by last feature column which we analysed above, then gets
                % the class labels for each biggest reduction row
                [ps,n] = sortrows(node.examples,winning_feature);
                ls = node.labels(n);

                % Set current feature which had the biggest reduction in
                % impurity as current node, setting splitFeature as feature column index,
                % splitFeatureName as the feature name and value that it should be split by
                node.splitFeature = winning_feature;
                node.splitFeatureName = m.feature_names{winning_feature};
                % Choose split feature value as halfway between biggest reduction feature and above, accounting for space inbetween
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

                % Can remove the examples and labels for training data as
                % they will now move down to live in nodes children, Also
                % removing the prediction as won't need one for split node only leaf node
                % Removing saves deplicated data
                node.examples = [];
                node.labels = [];
                node.prediction = [];

                % Create two child nodes coming off current node from empty
                % node static data created above


                node.children{1} = m.emptyNode;
                m.nodes = m.nodes + 1;          % Increment number of nodes value
                node.children{1}.number = m.nodes;    % Set which node number(unique) it is
                % First child has rows that are from 1 to the index with
                % the biggest reduction in impurity, setting the examples
                % and labels from the sorted data above
                node.children{1}.examples = ps(1:winning_index,:);
                node.children{1}.labels = ls(1:winning_index);
                % Finds the most common class label value from the labels
                node.children{1}.prediction = mode(node.children{1}.labels);

                % Same as above but for child 2, and setting the examples
                % and labels this time from between the biggest reduction
                % in impurity value + 1 and the end
                node.children{2} = m.emptyNode;
                m.nodes = m.nodes + 1;
                node.children{2}.number = m.nodes;
                node.children{2}.examples = ps((winning_index+1):end,:);
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);

                % Recursive call to same/current function, checking if
                % both the children can split, does until they cannot be
                % split any more
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end

        end

        %% Finds the class label impurity, this allows us to gain confidence in our predictions, done using GDI(Gini's diversity index)
        %% If a node is 0 it is pure and will have all class labels the same, if a node has a random mixture of different class levels it is said to have high impurity
        % Parameters: m model data structure with model attributes,
        % current class labels for node
        % Returns: the GDI(measure of impurity)
        function e = weightedImpurity(m, labels)

            % Define weight which is used to rescale the GDI score, it
            % allows for fair comparisons between different numbers of
            % nodes with different sizes
            weight = length(labels) / m.N;

            summ = 0;

            % Find how many class labels there are in the set, used for the
            % p in GDI equation
            obsInThisNode = length(labels);

            % Loop over the amount of unique class labels
            for i=1:length(m.unique_classes)

                % Calculate a fraction which represents how many of a
                % certain class label is in the whole set
				pc = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                summ = summ + (pc*pc);      % Square the fraction and add it to the overall sum(Sigma)

            end

            % Finds GDI g, formula is 1 - sum of fraction of class labels
            % in set which belong to that class squared
            g = 1 - summ;

            % Multiply the GDI by the weight to allow fair comparison
            % between different numbers of nodes with different sizes
            e = weight * g;

        end

        %% Responsible for processes that make a prediction for test examples, using the trained model
        % Parameters: trained decision tree model m, test data examples table
        % without class label
        % Returns: array of class label predictions for test examples
        function predictions = predict(m, test_examples)

            predictions = categorical;

            % Iterate over each test data example indexes
            for i=1:size(test_examples,1)

				% fprintf('classifying example %i/%i\n', i, size(test_examples,1));

                % Get test data example as one row array, pass it through
                % with the trained model to predict_one to gets it predicted class label
                this_test_example = test_examples{i,:};
                this_prediction = mytree.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;   % Add predicted value to predictions array

			end
        end

        %% Finds the predicted class label for test data example row using trained model, calling recursive function descend_tree to search tree
        % Parameters: trained decision tree model m, test data example row
        % without class label
        % Returns: predicted class label for test data example row
        function prediction = predict_one(m, this_test_example)

            % Call recursive function to search decision tree until gets
            % predicted value
			node = mytree.descend_tree(m.tree, this_test_example);

            % Sets class label prediction of the winning leaf nodes
            % prediction
            prediction = node.prediction;

		end

        %% Recursive function to descent tree, applying the splitting rules for each node with the test data example
        % Parameters: current node in the recusive cycle starts with the
        % root node on the tree, test data example row without class label
        % Returns: current node in tree with are checking whether to
        % descend on or return as the winning leaf node
        function node = descend_tree(node, this_test_example)

            % Checks if the node has any children, if not then its is the
            % winning leaf node and is returned from recursive function
			if isempty(node.children)
                return;
            else

                % Comparing the test data example feature value by whether
                % it is less than the split value, if it is returns the
                % 1st child, if not returns the 2nd due to being a greater value
                if this_test_example(node.splitFeature) < node.splitValue
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                else
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end

		end

        %% Recursive function that prints a statement descibing the layout of the decision tree
        % Parameters: decision tree that has been made from trained data
        function describeNode(node)

			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});
            end

		end

    end
end
