function nonDetLabels = nonDeter(states, regions, regionNames, simRel,DFA)
    % Get the number of actions and states
    num_actions = numel(DFA.act);
    num_states = numel(states);

    % Initialize the nonDetLabels matrix with zeros
    nonDetLabels = zeros(num_actions, num_states);

    % Define epsilon
    epsilon = simRel.epsilon;

    % Iterate over each state in states
    for i = 1:num_states
        state = states(i);

        % Define the bounds of the enlarged state
        enlarged_min = state - epsilon;
        enlarged_max = state + epsilon;

        % Determine which regions the enlarged state intersects
        intersected_regions = false(1, numel(regions));
        for j = 1:numel(regions)
            region_min = min(regions(j).V);
            region_max = max(regions(j).V);
            
            % Check if the enlarged state intersects the region
            if enlarged_min <= region_max && enlarged_max >= region_min
                intersected_regions(j) = true;
            end
        end

        % Find the corresponding action index for the combination of intersected regions
        if any(intersected_regions)
            % Create a string representing the combination of intersected regions
            intersected_labels = regionNames(intersected_regions);
            intersected_action = strjoin(sort(intersected_labels), '');

            % Find the index of this action in DFA.act
            action_index = find(strcmp(DFA.act, intersected_action), 1);
            
            if ~isempty(action_index)
                % Mark the intersection in the nonDetLabels matrix
                nonDetLabels(action_index, i) = 1;
            end
        end
    end
    zero_columns = all(nonDetLabels == 0);

% Set the first element of these columns to 1
    nonDetLabels(1, zero_columns) = 1;
end
