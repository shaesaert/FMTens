function matrix = createDescendingMatrix(n)
    % Generate the vector with elements from n^2 to 1
    elements = n^2:-1:1;
    
    % Reshape the vector into an n x n matrix
    matrix = reshape(elements, [n, n])';
end
