function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------
% add ones to X data matrix
% size(X) = 5000x400
X = [ones(m,1) X];

% calculate activation (hidden layer)
% size(Theta1) = 25x401
% size(a)      = 5000x25
a = sigmoid(X * Theta1');

% add ones to activation matrix of hidden layer
a = [ones(size(a, 1), 1) a];

% hypothesis output
% size(Theta2) = 10x26
% size(hx)     = 5000 x 10
hx = sigmoid(a * Theta2');

for label=1:num_labels
    % convert y to logic array for each label
    y_t = (y == label);

    % calculate cost for each label and add to overall cost
    J = J - (y_t' * log(hx(:, label)) + (1 - y_t)' * log(1 - hx(:, label))) / m;
end

% remove the 1st column of each theta
Theta1_t = Theta1(:, 2:end);
Theta2_t = Theta2(:, 2:end);
% unrolled theta without 1st row
theta = [Theta1_t(:); Theta2_t(:)];

% regularization computation
reg = theta' * theta;

J = J + reg * lambda / (2 * m);

% -------------------------------------------------------------
% a3 = hx;
% a2 = a;
% delta3 = delta_hx
% delta2 = delta_a
for label=1:num_labels
    y_t = (y == label);

    delta_hx(:, label) = (hx(:, label) - y_t);
end

% size(Theta2_t) = 10x25
% size(delta_hx) = 5000x10
% size(a2)       = 5000x25
a2 = a(2:size(a,2), :);
delta_a = (delta_hx * Theta2_t) .* sigmoidGradient(a2);

delta = 0;
X = X(2:size(X,2), :);
% size(delta_a) = 5000x25;
% size(X)       = 5000x1;
delta = delta + X' * delta_a;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
