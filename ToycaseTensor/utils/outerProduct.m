function result = outerProduct(varargin)
    % This function computes the outer product of N given column vectors.
    % Input: varargin - variable number of column vectors
    % Output: result - the outer product tensor

    % Number of input vectors
    numVectors = nargin;
    
    % Validate inputs
    if numVectors < 2
        error('At least two vectors are required.');
    end
    
    % Get the size of each vector
    sizes = cellfun(@length, varargin);
    
    % Initialize the result with the first vector
    result = varargin{1};
    
    % Compute the outer product iteratively
    for i = 2:numVectors
        vec = varargin{i};
        result = bsxfun(@times, reshape(result, [], 1), reshape(vec, 1, []));
        newSize = [sizes(1:i-1), sizes(i)];
        result = reshape(result, newSize);
    end
end
