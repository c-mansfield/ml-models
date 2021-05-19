% Object to train and predict data label values using K-NN(Nearest Neighbour)

classdef myknn
    methods(Static)
        
        %% Processes that trains K-NN model from training dataset
        % Parameters: dataset of training data examples without class
        % labels, array of training data class labels, value of nearest
        % neighbours(K) to check 
        % Returns: static data object m with all trained K-NN attributes
        function m = fit(train_examples, train_labels, k)
            
            % Start of Z-score standardisation - Make sure no big attribute ranges
            % Finds the average of all attributes(mean) and what the spread of the attributes 
            % are(standard deviation) of training data for z-score, stores in data struct as mean and std
			m.mean = mean(train_examples{:,:});
			m.std = std(train_examples{:,:});
			
            % Loop through index of each training data example
            for i=1:size(train_examples,1)
                % Subtract the average of all attributes from each attribute for each row in training example data
				train_examples{i,:} = train_examples{i,:} - m.mean;
				
                % Divide by how spread out all the attributes are for each attribute in each
                % row in the training example data
                train_examples{i,:} = train_examples{i,:} ./ m.std;
				
            end
            
            m.train_examples = train_examples;
            m.train_labels = train_labels;
            m.k = k;
        
        end
        
        %% Predicts class labels of data from trained K-NN classifier dataset
        % Parameters: trained dataset, test dataset without class labels.
        % Returns: array of predicted class labels  
        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            % Loop through indexes of test data example rows
            for i=1:size(test_examples,1)
                
                %fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                % Gets current row in testing data examples from index
                this_test_example = test_examples{i,:};
                
                % Start Z-score standardisation for the prediction data
                % Subtracts the mean of the original training data from
                % each attribute in each prediction example data row
                this_test_example = this_test_example - m.mean;
				
                % Divides by standard deviation for each attribute in each
                % prediction example data row
                this_test_example = this_test_example ./ m.std;
				                
                % Finds the predicted class label for current prediction data
                % example in iteration
                this_prediction = myknn.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
            end
        
		end
        
        %% Processes to predict class label for one row of prediction data
        % Parameters: trained dataset, row from test data that is getting
        % class label predicted.
        % Returns: predicted class label from test data row & trained data
        function prediction = predict_one(m, this_test_example)
            
            % Finds all distances from points in training data to test data
            % point
            distances = myknn.calculate_distances(m, this_test_example);
			
            % Gets k number of training data array indexes for the smallest
            % distances
            neighbour_indices = myknn.find_nn_indices(m, distances);
			
            % Finds the most common class label in closest training data
            prediction = myknn.make_prediction(m, neighbour_indices);
        
        end
        
        %% Finds distances between each trained dataset row and testing example row
        % Parameters: trained dataset, row from test data that is getting
        % class label predicted.
        % Returns: Array of distances from each trained data row and
        % testing data example row
        function distances = calculate_distances(m, this_test_example)
            
			distances = [];
            
            % Loop through indexes of training data example rows
			for i=1:size(m.train_examples,1)
                
                % Gets full row from training example dataset for index i
				this_training_example = m.train_examples{i,:};
				
                % Finds the distance between training example point to test
                % data example point
                this_distance = myknn.calculate_distance(this_training_example, this_test_example);
                distances(end+1) = this_distance;
            end
        
		end
        
        %% Finds distance between two points using Euclidean distance
        % Parameters: trained data set row array, testing data example row
        % array
        % Returns: distance between the two points
        function distance = calculate_distance(p, q)
            
            % Subtracts each element in p array by adjacent value in q
            % array, adds each new value to 'differences' array
			differences = q - p;
			
            % Square each value in 'differences' array then adds each
            % new value to new array 'squares'
            squares = differences .^ 2;
            total = sum(squares);       % All values in 'squares' array added together
            distance = sqrt(total);     % Square roots total
        
		end
        
        %% Finds K(number of neighbours) of nearest points to testing data point
        % Parameters: trained dataset, distances from testing example row
        % point to each of the other points in the trained data.
        % Returns: array of indexes from the distances array of the
        % smallest distances.
        function neighbour_indices = find_nn_indices(m, distances)
            
            % Sorts distances from lowest to highest, stores output array indexes of
            % these distances in 'indices'
			[~, indices] = sort(distances);
			
            % Gets 1 to K(number of neighbours) of distance indexes, stores
            % in 'neighbour_indices' array
            neighbour_indices = indices(1:m.k);
        
		end
        
        %% Predicts value of class label based on the most common class label of neighbouring points
        % Parameters: trained dataset, indexes of nearest row of data in
        % trained data.
        % Returns: predicted class label of testing dataset example
        function prediction = make_prediction(m, neighbour_indices)
            
            % Gets all training data class labels which index matches values
            % in 'neighbour_indices' array, stores as array 'neighbour_labels'
			neighbour_labels = m.train_labels(neighbour_indices);
			
            % Finds most common label in 'neighbour_labels' array
            prediction = mode(neighbour_labels);
        
		end

    end
end