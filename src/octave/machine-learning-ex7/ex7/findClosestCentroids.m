function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X, 1);

for i = 1:m
    X_i = X(i, :);

    % distance to the 1st centroid
    tmp = X_i - centroids(1, :);
    min_distance = tmp * tmp';
    idx(i) = 1;

    % calculate distance from X(i) to every centroids
    % and get index of the closest one
    for j = 2:K
        tmp = X_i - centroids(j, :);
        distance = tmp * tmp';
        % store the mininum distance centroid index
        if (distance < min_distance)
            idx(i) = j;
            min_distance = distance;
        end
    end
end
% =============================================================

end

