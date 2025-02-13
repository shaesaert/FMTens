function [rowIndices, colIndices] = findElementIndices(matrix)
    % Initialize the matrix size
    [numRows, numCols] = size(matrix);

    % Preallocate arrays for row and column indices
    rowIndices = zeros(1, numRows * numCols);
    colIndices = zeros(1, numRows * numCols);

    % Find the indices of each element
    for value = 1:(numRows * numCols)
        [row, col] = find(matrix == value);
        rowIndices(value) = row;
        colIndices(value) = col;
    end
end
