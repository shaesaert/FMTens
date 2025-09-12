function [formula, sys] = overtakeSpec(sys)
%%% Function to define specification using scLTL
%
% Input
% -----
% sys = dynamicla system to evaluate
%
% Output
% -----
% formula = Formulated scLTL specification
% sys = Updated dynamical system including the
%       atomic proposition and polytope regions
%

    % Vector to extract the values from all the states
    xe_xo = [1,0,0,0];
    ye_yo = [0,1,0,0];

    y_thres = 1.8;
    safe_dist = 5;

    % Goal of the vehicle
    P1 = Polyhedron('A', [ye_yo;-ye_yo], 'b', [y_thres;y_thres] ); % Ego vehicle in left lane
    P2 = Polyhedron('A',-xe_xo,'b',-safe_dist);
    goal = "(p1 & p2)";

    % Keep safe x-distance (front and back)
    P3 = Polyhedron('A',xe_xo,'b',-safe_dist); % Rear distance
    x_dist = "(!p1 | (p3 | p2))";
        
    sys.regions = [P1; P2; P3];
    sys.AP = {'p1','p2','p3'};
    formula_str = x_dist + "U (" + goal + " & " + x_dist + ")";
    formula = convertStringsToChars(formula_str);

end