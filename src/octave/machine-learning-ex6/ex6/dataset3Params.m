function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = C_vec;

m = length(C_vec);
error = zeros(m, m);

for i = 1:m
    C = C_vec(i);
    for j = 1:m
        sigma = sigma_vec(j);
        # train model for every pair of C & sigma
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        
        # calculate error of each prediction set
        error(i, j) = mean(double(predictions ~= yval));
    end
end

min_error = error(1, 1);
C = C_vec(1);
sigma = sigma_vec(1);

# get result of least error pair of C & sigma
for i = 1:m
    for j = 1:m
        if (error(i, j) < min_error)
            C = C_vec(i);
            sigma = sigma_vec(j);
            min_error = error(i, j);
        end
    end
end
% =========================================================================

end
