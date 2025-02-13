clc;
close all;
clear all;

%% system dynamics

% Define the state and input space parameters
state_min = -20;
state_max = 5;
num_state_grids = 100;

input_min = -5;
input_max = 5;
num_input_grids = 10;

% Generate state space grid centers
state_grid_centers = linspace(state_min, state_max, num_state_grids);

% Generate input space grid centers
input_grid_centers = linspace(input_min, input_max, num_input_grids);

% Initialize the probability matrix
P = zeros(num_state_grids, num_state_grids);

% For each state and each input, compute the next state
for i = 1:num_state_grids
    for j = 1:num_input_grids
        % Current state
        current_state = state_grid_centers(i);
        % Input
        input = input_grid_centers(j);
        % Compute next state
        next_state = current_state + input;
        
        % Find the nearest state grid center
        [~, closest_idx] = min(abs(state_grid_centers - next_state));
        
        % Increment the transition probability
        P(i, closest_idx) =1+P(i, closest_idx);
    end
end

% Normalize the rows of the probability matrix to sum to 1
for i = 1:num_state_grids
    P(i, :) = P(i, :) / sum(P(i, :));
end

% Display the probability matrix
disp('Transition Probability Matrix P:');
disp(P);



%% tensor, without policy
n = [10 10 10];


tic;
% Piss row: next state, column: current state
%% dimension 1
% u1 P1ss(s',s) (+ direction 0.8, disturbance= opposite direction 0.2)
P1ss_u1 = zeros(n(1),n(1));
for i = 1:size(P1ss_u1,2)
    if i+1 >=1 & i+1<= size(P1ss_u1,2)
        P1ss_u1(i+1,i) = 0.8;
    else
    end

    if  i-1 >=1 & i-1 <= size(P1ss_u1,2) 
        P1ss_u1(i-1,i) = 0.2;
    else
    end
    i = i+1;
end

% STAY u2. P1ss(s',s)  (no input no disturbance)
P1ss_u2 = zeros(n(1),n(1));
for i = 1:size(P1ss_u2,2)

        P1ss_u2(i,i) = 1;


    % if  i-1 >=1 & i-1 <= size(P1ss_u2,2) 
    %     P1ss_u2(i-1,i) = 0.2;
    % else
    % end
    i = i+1;
end

% weighted policy 0.8 actioning on going right, 0.2 actioning on staying
P1ss = 0.8*P1ss_u1+0.2*P1ss_u2;


%% dimension 2
% u1. P2ss_u1(s',s)  (+ direction 0.8, disturbance= opposite direction 0.2)
P2ss_u1 = zeros(n(2),n(2));
for i = 1:size(P2ss_u1,2)
    if i+1 >=1 & i+1<= size(P2ss_u1,2)
        P2ss_u1(i+1,i) = 0.8;
    else
    end

    if  i-1 >=1 & i-1 <= size(P2ss_u1,2) 
        P2ss_u1(i-1,i) = 0.2;
    else
    end
    i = i+1;
end

% STAY u2: P2ss_u2(s',s)  (no input no disturbance)
P2ss_u2 = zeros(n(2),n(2));
for i = 1:size(P2ss_u2,2)

        P2ss_u2(i,i) = 1;


    % if  i-1 >=1 & i-1 <= size(P2ss_u2,2) 
    %     P2ss_u2(i-1,i) = 0.2;
    % else
    % end
    i = i+1;
end

% weighted policy 0.8 actioning on going up, 0.2 actioning on staying
P2ss = 0.8*P2ss_u1+0.2*P2ss_u2;

%% dimension 3
% u1. P2ss_u1(s',s)  (+ direction 0.8, disturbance= opposite direction 0.2)
P3ss_u1 = zeros(n(3),n(3));
for i = 1:size(P3ss_u1,2)
    if i+1 >=1 & i+1<= size(P3ss_u1,2)
        P3ss_u1(i+1,i) = 0.8;
    else
    end

    if  i-1 >=1 & i-1 <= size(P3ss_u1,2) 
        P3ss_u1(i-1,i) = 0.2;
    else
    end
    i = i+1;
end

% STAY u2: P2ss_u2(s',s)  (no input no disturbance)
P3ss_u2 = zeros(n(3),n(3));
for i = 1:size(P3ss_u2,2)

        P3ss_u2(i,i) = 1;


    % if  i-1 >=1 & i-1 <= size(P2ss_u2,2) 
    %     P2ss_u2(i-1,i) = 0.2;
    % else
    % end
    i = i+1;
end

% weighted policy 0.8 actioning on going up, 0.2 actioning on staying
P3ss = 0.8*P3ss_u1+0.2*P3ss_u2;

Time_Pss = toc;


Pss_cell = cell(numel(n),1);
Pss_cell{1} = P1ss;
Pss_cell{2} = P2ss;
Pss_cell{3} = P3ss;

%% Value function
gamma = 0.99999; % discount factor, each action
max_iterations = 200; % Maximum number of iterations

tic;
Vcell = cell(numel(n),1);
for i = 1:numel(n)
    Vcell{i} = zeros(n(i),1);
    Vcell{i}(round(0.5*n(i))) = 1;
    Vcellupdate = highDimensionVI(gamma,max_iterations,Vcell{i},Pss_cell{i});
    Vcell{i} = Vcellupdate;
    i = i+1;
end

V = outerProduct(Vcell{:});
Time_VI = toc;


% %% visualization 2D
% figure;
% Vvis = flipud(V');
% imagesc(Vvis);
% imagesc(V);
% colorbar;
% 
% xlabel('x_1 direction');
% ylabel('x_2 direction');
% 
% 
% % xticks(1:size(V, 2));
% % yticks(1:size(V, 1));
% title('Visualization of V with Colorbar');
% 
% 
% axis equal;



%% visualiza 3D
tic;
Vvis = V;
for i = 1:numel(n(3))
    Vvis(:,:,i) = flipud(Vvis(:,:,i)');
    i = i+1;
end

% Vvis = permute(flipud(permute(Vvis, [1 2 3])), [1 2 3]);

x = 1:size(Vvis, 1);
y = 1:size(Vvis, 2);
z = 1:size(Vvis, 3);

% Create meshgrid for 3D plotting
[X, Y, Z] = meshgrid(x, y, z);

% Plotting
figure;
slice(X, Y, Z, Vvis, [], [], z);
xlabel('x axis');
ylabel('y axis');
zlabel('z axis');
title('Visualization of the 3D V in 3D state space');
colorbar;
Time_vis = toc;

disp(['Elapsed time for Pss: ' num2str(Time_Pss) ' seconds']);
disp(['Elapsed time for Value iteration: ' num2str(Time_VI) ' seconds']);
disp(['Elapsed time for Visualization: ' num2str(Time_vis) ' seconds']);



