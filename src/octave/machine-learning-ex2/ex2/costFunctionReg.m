function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis function values
hx = sigmoid(X * theta);

% regularization component
% sum of theta except 1st element could be calculated by:
%   reg = sum(theta.^2) - theta(1) ^ 2;
% or:
%   reg = theta' * theta - theta(1) ^ 2;
%
reg = (theta' * theta - theta(1) ^ 2);
reg = reg * lambda / (2 * m);

% cost function
J = -(y' * log(hx) + (1 - y)' * log(1 - hx)) / m;
J = J + reg;

% gradient (see costFunction.m)
grad = X' * (hx - y) / m;

% regularization component for gradient
grad_reg = theta * lambda / m;
grad_reg(1) = 0;    % the 1s element is 0

% gradient with regularization component
grad = grad + grad_reg;

% =============================================================

end
