clc;
clear all;
close all;

tic;
%% define system dynamics
A = [1 1 1];
B = [1 1 1];
C = [1 1 1];
D = [0 0 0];
Bw = [sqrt(0.25) sqrt(0.25) sqrt(0.25)];
dim = numel(A);

mu = zeros(1, 1,1); % Mean of disturbance
sigma = eye(1); % Variance of disturbance

xl = -20;
xu = 5;
l = 10;
tol=10^-3;  

ul = -5;
uu = 5;
lu = 10; 

sysLTI = cell(numel(A),1);
uhat = cell(numel(A),1);
sysAbs = cell(numel(A),1);
P = cell(numel(A),1);
tensorP = cell(numel(A),1);

for i=1:numel(A)
    sysLTI{i} = LinModel(A(i), B(i), C(i), D(i), Bw(i), mu, sigma);
    sysLTI{i}.X = Polyhedron(combvec([xl,xu])');
    sysLTI{i}.U = Polyhedron(combvec([ul,uu])');
    uhat{i} = GridInputSpace(lu,sysLTI{i}.U,'log');
    sysAbs{i} = tensorAbs(sysLTI{i},uhat{i},l,tol,'TensorComputation',true);
    P{i} = sysAbs{i}.P.Prob;
    nu = size(uhat{i},2);
    tensorP{i} = zeros(l,nu,l);
    for j = 1:nu
        submatrix = P{i}(:, (j-1)*l+1:j*l);
        tensorP{i}(:, j, :) = permute(submatrix, [1, 3, 2]);
    end
    tensorP{i}= tensorP{i} ./ sum(tensorP{i}, 1); 
end

TimeSysAbs = toc;

tic;
gamma = 0.6;
V = cell(numel(A),1);
pi = cell(numel(A),1);
maxIterations = 1000;
tolerance = 1e-6;

for i = 1:numel(A)
    V{i} = zeros(l,1);
    i = i+1;
end

V{1}(l) = 1;
V{2}(l) = 1;
V{3}(l) = 1;

for i = 1:numel(A)
    pi{i} = zeros(l,nu);

    fixedStates = (V{i}==1);

    for iter = 1:maxIterations
        V_old = V{i};

        for k = 1:l
            if fixedStates(k)
                continue;
            end

            Q = zeros(nu,1);

            for m = 1:nu
                %Q(m) = squeeze(tensorP{i}(:, m, k))' * V_old;
                Q(m) = sum(tensorP{i}(:,m,k).*V_old);
                
                %Q(m) = squeeze(tensorP{i}(k, m, :))' * V_old;
            end

            V{i}(k) = gamma*max(Q);

            optimalControls = (Q == max(Q));
            pi{i}(k,:) = optimalControls / sum(optimalControls);
        end

        if max(abs(V{i}-V_old)) < tolerance
            break;
        end
       
    end

    
    i = i+1;
    
end

Vtensor = outerProduct(V{1},V{2},V{3});
Time_VI = toc;

tic;

tensor = Vtensor;

%%%%%%%%
% Define the grid and colormap
[x, y, z] = meshgrid(1:10, 1:10, 1:10);
cmap = [linspace(0, 1, 64)', zeros(64, 1), linspace(1, 0, 64)'];
colormap(cmap);
caxis([0, 1]);

% Plotting each xy plane as a 3D slice
figure;
hold on;

for k = 1:10
    % Create a surface for each xy plane
    surf(squeeze(tensor(:, :, k)), 'EdgeColor', 'k');
end

xlabel('x axis');
ylabel('y axis');
zlabel('z axis');
title('3D Visualization of Tensor Slices with Grid Lines');
colorbar;
view(3); % 3D view
axis tight;
grid on;

TimeVisualization = toc;
%%%%%%%%%%%%%%

tensor = Vtensor;
% Define the grid
[x, y, z] = meshgrid(1:10, 1:10, 1:10);

% Create a custom colormap: From blue to red (0 to 1)
cmap = [linspace(0, 1, 64)', zeros(64, 1), linspace(1, 0, 64)'];
colormap(cmap);

% Set color limits to ensure the value 1 corresponds to red
caxis([0, 1]);

% Plotting slices of the tensor
figure;
slice(x, y, z, tensor, [], [], 1:10);  % Slicing along the z axis
xlabel('x axis');
ylabel('y axis');
zlabel('z axis');
title('Slices of a 10x10x10 Tensor');
colorbar;

% Adjust view
view(3);  % 3D view

% % Add lighting and shading for better visualization
shading interp;
lighting phong;
camlight right;

TimeVisualization = toc;

d = whos();
d= sum([d.bytes])
memory = d*10^(-6);

fprintf('Memory usage in total is: %.2f MB \n', memory);
fprintf('Time used for system definition and abstraction on the decoupled system is %.2f seconds\n', TimeSysAbs);
fprintf('Time used for value iteration on the decoupled system is %.2f seconds\n', Time_VI);
fprintf('Time used for visualization is %.2f seconds\n', TimeVisualization);


