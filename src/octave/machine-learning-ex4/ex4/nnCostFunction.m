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

labels = (1:num_labels);

% Part 1: Feedforward
%------------------------------------------------
X = [ones(m, 1) X];

% X: [5000x401], Theta1: [25x401], a: [5000x25]
a = X * Theta1';
a = sigmoid(a);

% a: [5000x26], Theta2: [10x26], hx: [5000x10]
a = [ones(size(a, 1), 1) a];
hx = sigmoid(a * Theta2');

for label = 1:num_labels
    y_t = (y == label);

    J = J - (y_t' * log(hx(:, label)) + (1 - y_t') * log(1 - hx(:, label))) / m;
end

Theta1_t = Theta1(:, 2:end);
Theta2_t = Theta2(:, 2:end);
theta = [Theta1_t(:); Theta2_t(:)];
reg = theta' * theta;

J = J + reg * lambda / (2 * m);

% Part 2: Backpropagation
% -------------------------------------------------------------
% y_s = [1 2 3 4 5 6 7 8 9 10];
y_s = (1:num_labels);

for t = 1:m
    % step 1
    % a_1: [1x401], a_2: [1x26], a_3: [1x10]
    a_1 = X(t, :);
    a_2 = sigmoid(a_1 * Theta1');
    a_2 = [ones(size(a_2, 1), 1) a_2];
    a_3 = sigmoid(a_2 * Theta2');

    % step 2
    % delta_3: [1x10]
    y_t = (y(t) == y_s);
    delta_3 = a_3 - y_t;

    % step 3
    % dgz_2: [1x25], Theta2_t: [10x25], delta_2: [1x25]
    dgz_2 = a_2 .* (1 - a_2);
    dgz_2 = dgz_2(2:end);
    delta_2 = (delta_3 * Theta2_t) .* dgz_2;

    % step 4
    Theta1_grad = Theta1_grad + delta_2' * a_1;
    Theta2_grad = Theta2_grad + delta_3' * a_2;
end

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% step 5
grad = grad / m;

% Part 3: Regularization
% -------------------------------------------------------------
Theta1_reg = Theta1;
Theta2_reg = Theta2;

Theta1_reg(:, 1) = 0;
Theta2_reg(:, 1) = 0;

% unroll regularized params
Theta_reg = [Theta1_reg(:); Theta2_reg(:)];

% Regularized gradient
grad = grad + Theta_reg * lambda / m;

% =========================================================================

end
