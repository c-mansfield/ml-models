% Object to train and predict data label values using Naive Bayes model

classdef mynb
    methods(Static)

        %% Processes to train data using the naive bayes model, using normal distribution to get probability of a data class label
        % Parameters: dataset of training data examples without class
        % labels, array of training data class labels
        % Returns: static data with all trained Naive bayes attributes
        function m = fit(train_examples, train_labels)

            % Find all the different possiblities for the class label and
            % how many there are so we can work out the mean and standard
            % deviation for each column which have the same class label for the probability density function
            m.unique_classes = unique(train_labels);
            m.n_classes = length(m.unique_classes);

            % Initiliase cells for mean and standard deviation values
            m.means = {};
            m.stds = {};

            % Iterate through the length of class labels in the data
            for i = 1:m.n_classes

                % Finds the mean and the standard deviation for each row
                % with the same unique class label. These will be plugged into
                % the probability density function to get the Normal
                % distribution shape, normal distribution is never 0
				this_class = m.unique_classes(i);
                examples_from_this_class = train_examples{train_labels==this_class,:};
                m.means{end+1} = mean(examples_from_this_class);
                m.stds{end+1} = std(examples_from_this_class);

			end

            % Initiliase array which stores estimate of likeliness that
            % class label is to occur, based on how many times it is present
            % in the data. This is used later on when calculating a prediction
            m.priors = [];

            % Iterate through the length of class labels in the data
            for i = 1:m.n_classes

                % Finds a decimal(percentage) from dividing the current number
                % of examples with a class label by the total number of class labels.
                % Used in Bayes theorem formula to get probability that row
                % is class label, PRIOR
				this_class = m.unique_classes(i);
                examples_from_this_class = train_examples{train_labels==this_class,:};
                m.priors(end+1) = size(examples_from_this_class,1) / size(train_labels,1);
                s = size(train_labels,1);

			end

        end

        %% Splits the test data examples up into each row to find predictions for each rows class label
        % Parameters: naive bayes trained static data object m, example
        % data table from testing data wish to predict class labels for
        % Returns: array of predicted class labels of the test_examples
        function predictions = predict(m, test_examples)

            predictions = categorical;

            for i=1:size(test_examples,1)

				%fprintf('classifying example %i/%i\n', i, size(test_examples,1));

                % Get test_examples row from index for current iteration, pass it through
                % to the predict_one to get the class label prediction for
                % current row, add to predictions categorical array
                this_test_example = test_examples{i,:};
                this_prediction = mynb.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;

			end
        end

        %% Processes to predict the class label for test data examples row from the trained data
        % Parameters: naive bayes trained static data object m, row from
        % example test data we wish to predict class label for
        function prediction = predict_one(m, this_test_example)


            for i=1:m.n_classes

                % Finds the likelihood for the Bayes Theorem, the probibility a
                % feature has a value, given that it comes from a class
				this_likelihood = mynb.calculate_likelihood(m, this_test_example, i);

                % Gets the prior calculated in the 'fit' function for
                % current class label, used to calcualate Bayes Theorem
                this_prior = mynb.get_prior(m, i);

                % Calculates the probability for the class label, multiplying the prior and the
                % likelihood, BAYES THEOREM
                posterior_(i) = this_likelihood * this_prior;

			end

            % Gets the maximum value and its index from the probabilitys
            % for each class label, then with the highest probability
            % index can get the class label it represents
            [winning_value_, winning_index] = max(posterior_);
            prediction = m.unique_classes(winning_index);

        end

        %% Processes to get the likelihood for the bayes theorem formula, multiplying each feature together due to class-conditional independence assumption
        % Parameters: naive bayes trained static data object m, row from
        % example test data we wish to predict class to get likelihood,
        % class label index
        % Returns: Likelihood
        function likelihood = calculate_likelihood(m, this_test_example, class)

			likelihood = 1;

            % Iteration looping through the index of each test-data
            % example row, then multiplies probability density for each row
            % in test-data examples by one another to get the likelihood
			for i=1:length(this_test_example)

                % Calculate probability density curve for each individual test example
                % and multiplying each together to get the likelihood, class-conditional independent
                % assumption(considers each feature to be independant)
                likelihood = likelihood * mynb.calculate_pd(this_test_example(i), m.means{class}(i), m.stds{class}(i));

            end


        end

        %% Gets the prior(probability) of current class label, calculated in 'fit'
        % Parameters: naive bayes trained static data object m, current
        % class label index
        % Returns: prior value calculated in 'fit' function
        function prior = get_prior(m, class)

			prior = m.priors(class);

		end

        %% Calculates the probability density for current row using the mean and standard deviation, gets the normal distribution and how likely test example occurs
        % Parameters: x is a test example feature, mu is the mean of
        % controlling where the normal distribution is centered, sigma is
        % the standard deviation controlling the width
        % Returns: probability density(normal distribution curve) for
        % current data feature
        function pd = calculate_pd(x, mu, sigma)


            % Probality Density Formula. Same as:
            % normpdf()

			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);

		end

    end
end
