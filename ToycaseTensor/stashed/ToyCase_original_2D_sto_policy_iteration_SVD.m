clc;
clear all;
close all;


tic;
% System Definition
A = eye(2);
B = eye(2);
C = eye(2);
D = zeros(2);
Bw = sqrt(0.25) * eye(2);
dim = length(A);

mu = zeros(dim, 1); % Mean of disturbance
sigma = eye(dim); % Variance of disturbance

sysLTI = LinModel(A, B, C, D, Bw, mu, sigma);

x1l = -20;
x1u = 5;
x2l = -20;
x2u = 5;
sysLTI.X = Polyhedron(combvec([x1l, x1u], [x2l, x2u])');

ul = [-5; -5];
uu = [5; 5];
sysLTI.U = Polyhedron(combvec([ul(1), uu(1)], [ul(2), uu(2)])');

tol = 1e-6;

lu = 60; % number of abstract inputs in each direction
uhat = GridInputSpace(lu, sysLTI.U, 'log');
nu = size(uhat, 2); % [lu, nu] = [60, 3481];

l = [10 10];
nx = l(1) * l(2);

sysAbs = tensorAbs(sysLTI, uhat, l, tol, 'TensorComputation', true);
xhat = sysAbs.states;
% Phd = P_in_high_dimension_tensor(sysAbs,l,lu);

P = full(sysAbs.P.P_det); % size

tensorP = zeros(nx, nu, nx);
for i = 1:nu
    % Extract the i-th submatrix from P
    start_col = (i - 1) * nx + 1;
    end_col = i * nx;
    submatrix = P(:, start_col:end_col);

    % Assign the submatrix to the corresponding slice in tensorP
    tensorP(:, i, :) = submatrix;
end

P = tensorP;
TimeSysAbs = toc;

tic;

% Policy Iteration
gamma = 0.6;
max_iter = 1000;
threshold = 1e-9; % convergence threshold

% Initialize policy uniformly
pi = ones(nx, nu) / nu; % Uniform random policy

% Policy Iteration
policy_stable = false;
V = zeros(nx, 1);
V(nx) = 1; % Set the target state value to 1


b = rank(V);
while ~policy_stable
    % Policy Evaluation
    for iter = 1:max_iter

            V_prev = V;




        for i = 1:nx
            if i == nx
                continue; % Skip updating the goal state
            end
            V(i) = gamma * V_prev' * squeeze(P(i, :, :))' * pi(i, :)';
            V = min(V,1);
        end

        if max(abs(V - V_prev)) < threshold
            disp(['Policy Evaluation converged at iteration ' num2str(iter)]);
            break;
        end
    end

    % Policy Improvement
    policy_stable = true;
    for i = 1:nx
        if i == nx
            continue; % Skip updating the goal state
        end
        old_action = pi(i, :);
        Q = zeros(nu, 1);
        for u = 1:nu
            Q(u) = gamma * squeeze(P(i, u, :))' * V;
        end
        [~, best_action] = max(Q);
        pi(i, :) = 0; % Reset
        pi(i, best_action) = 1;
        if any(old_action ~= pi(i, :))
            policy_stable = false;
        end
    end
    %% rank reduction of V 
    r = rank(pi);
    V_reshaped = visualizeValueFunction(V,x1l,x1u,x2l,x2u,l);
    % V_reshaped_hat = cp_als(V_reshaped,2);
    [A,B,lam,V_reshaped_hat]=cp2_DCPD(V_reshaped,5);
    a= rank(V_reshaped);
    b=rank(V_reshaped_hat);
    V = reshape(V_reshaped_hat', 1, [])';
    
end
Time_PI = toc;
disp('Policy Iteration converged.');

%% Visualization
tic;
xLimits = [-20, 5];  % X-axis limits
yLimits = [-20, 5];  % Y-axis limits

% Calculate the number of grid points
numGridPointsX = l(1);
numGridPointsY = l(2);

stepSize = (x1u - x1l) / l(1);  % Step size for the grid

% Generate a grid of x and y coordinates
x = linspace(xLimits(1), xLimits(2), numGridPointsX);
y = linspace(yLimits(1), yLimits(2), numGridPointsY);

% Create a 2D grid of coordinates
[X, Y] = meshgrid(x, y);

% Reshape V to match the 2D grid
V_reshaped = reshape(V, [numGridPointsY, numGridPointsX]);

% Create the plot
figure;
Vvis = mat2gray(V_reshaped);
imagesc(x, y, Vvis);
colorbar;
xlabel('X-axis');
ylabel('Y-axis');
title('Visualization of the reachability probabilities iterated on the original system');
set(gca, 'YDir', 'normal');  % Set Y-axis to normal direction
TimeVisualization = toc;


d = whos();
d= sum([d.bytes])
memory = d*10^(-6);

fprintf('Memory usage in total is: %.2f MB\n', memory);
fprintf('Time used for system definition and abstraction on the original 2D system is %.2f seconds\n', TimeSysAbs);
fprintf('Time used for policy iteration on the original system (rank order reduction applied) is %.2f seconds\n', Time_PI);
fprintf('Time used for visualization is %.2f seconds\n', TimeVisualization);



