function V = highDimensionVI(gamma,max_iterations,V,Pss)

tolerance = 1e-6; % Tolerance for convergence

% Value iteration until convergence
delta = Inf;
iteration = 0;

while delta > tolerance && iteration < max_iterations
    delta = 0;
    indices = find(V ~= 0);
    row = ind2sub(size(V), indices); % two-dimensional state
% [row, col, page] = ind2sub(size(V), indices); % three-dimensional state
    for j = 1:numel(row)
    indices2 = find(Pss(row(j),:) ~= 0);
    row2 = ind2sub([size(Pss,2)], indices2);
    for i = 1:numel(row2)
        V_inter = V(row(j))*Pss(row(j),row2(i))*gamma;
        if V_inter >= V(row2(i))
           
            delta = max(delta, abs(V(row2(i)) - V_inter));
                V(row2(i)) = V_inter; % Update V if the new value is higher
        else
        end
        i = i+1;
     end
    j = j+1; % in case there are more than 1 positive-valued V at intialization
    end
    iteration = iteration + 1;
end
end