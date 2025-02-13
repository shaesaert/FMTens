function subVectors = divideAscendingSequences(v)
    % Initialize the cell array to hold sub-vectors
    subVectors = {};
    % Initialize the first sub-vector
    currentSubVector = v(1);

    % Iterate through the vector
    for i = 2:length(v)
        if v(i) >= v(i-1)
            % Continue the ascending sequence
            currentSubVector = [currentSubVector, v(i)];
        else
            % Store the current sub-vector and start a new one
            subVectors{end+1} = currentSubVector; %#ok<*AGROW>
            currentSubVector = v(i);
        end
    end
    % Store the last sub-vector
    subVectors{end+1} = currentSubVector;

    % % Display the result
    % for i = 1:length(subVectors)
    %     fprintf('v%d = [%s]\n', i, num2str(subVectors{i}));
    % end
end
