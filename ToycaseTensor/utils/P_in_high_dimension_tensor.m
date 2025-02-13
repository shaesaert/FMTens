function Phd = P_in_high_dimension_tensor(sysAbs,l,lu)
[row,col] = find(sysAbs.P.P_det);
subrow = divideAscendingSequences(row);
currentstate = subrow;

subcol = cell(1,numel(subrow));
subcol{1} = col(1:numel(subrow{1}));
subcol{1} = transpose(subcol{1}); % to align format with subrow

nrOfidx = numel(subcol{1});

for i = 2:numel(subrow)
    subcol{i} = col(nrOfidx+1:nrOfidx+numel(subrow{i}));
    nrOfidx = nrOfidx + numel(subcol{i});
    subcol{i} = subcol{i}- (i-1)*(l(1)*l(2));
    subcol{i} = transpose(subcol{i}); % to align format with subrow
    i = i+1;
end

nextstate = subcol;

Phd = zeros(l(1),l(2),lu-1,lu-1,l(1),l(2));
% each dimension: lower bound -> higher bound,
% dim1 : x axis of position
% dim2 : y axis of position
% dim3 : x axis of position
% dim4 : y axis of position
% dim5 : x axis of position
% dim6 : y axis of position
% assuming that each action has in total positive transition probability
% distribution
matrix = createDescendingMatrix(l(1)); % assuming l(1)=l(2)
[rowIndices, colIndices] = findElementIndices(matrix); % index of ascending states

%% deterministic transition
for i4 = 1:lu-1
    for i3 = 1:lu-1
        order_currentstate = currentstate{(i4-1)*(lu-1)+i3};
        order_nextstate = nextstate{(i4-1)*(lu-1)+i3};

        for k = 1:numel(order_currentstate)
            k_current_value = order_currentstate(k);
            k_next_value = order_nextstate(k);
            row_idx_current_state = rowIndices(k_current_value);
            col_idx_current_state = colIndices(k_current_value);
            row_idx_next_state = rowIndices(k_next_value);
            col_idx_next_state = colIndices(k_next_value);
            % [row_idx_current_state,col_idx_current_state] = [rowIndices(k_current_value),colIndices(k_current_value)];
            % [row_idx_next_state,col_idx_next_state] = [rowIndices(k_next_value),colIndices(k_next_value)];

            Phd(row_idx_current_state,col_idx_current_state,i3,i4,row_idx_next_state,col_idx_next_state) = 1;

            k = k+1;
        end


        i3 = i3+1;
    end
    i4 = i4+1;
end

end