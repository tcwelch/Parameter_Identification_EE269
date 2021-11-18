function [k_predictions] = knn(train_data,test_data, k, train_y,use_uniform)
% Takes in training data, test data, and k value to identify wn and zeta based on
% the k nearest neighbors for the test data.
% If use_uniform = 0, take weighted average of neighbors using inverse
% distance

% Find K nearest neighbors
[indices_top_k, distances] = knnsearch(train_data, test_data, 'K', k);
% Identify wn and zeta for each sample based on K nearest neighbors 
k_predictions = zeros(length(indices_top_k),2);
for i = 1:size(indices_top_k,1)
    feature_sum = 0;
    if(use_uniform)
        for j = 1:size(indices_top_k,2)
            feature_sum = feature_sum + train_y(indices_top_k(i,j),:);
        end
        k_predictions(i,:) = feature_sum / k;
    else
        for j = 1:size(indices_top_k,2)
            feature_sum = feature_sum + inv(distances(i,j))*train_y(indices_top_k(i,j),:);
        end
        k_predictions(i,:) = feature_sum / sum(distances(i,:).^-1); 
    end
end

