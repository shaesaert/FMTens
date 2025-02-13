clc;
clear all;
close all;

tic;
%% define system dynamics
A = [1 1]; %fix coding practice here better use cell structure
B = [1 1 ];
C = [1 1];
D = [0 0 ];
Bw = [sqrt(0.25) sqrt(0.25)];
dim = numel(A);

mu = zeros(1, 1); % Mean of disturbance
sigma = eye(1); % Variance of disturbance


sysLTI = cell(numel(A),1);
uhat = cell(numel(A),1);
sysAbs = cell(numel(A),1);
P = cell(numel(A),1);
tensorP = cell(numel(A),1);

xl = -20;
xu = 5;
l = 560;
tol=10^-3;  

ul = -5;
uu = 5;
lu = 14; 


%% Step 1 Translate the specification
% 1) pick up region
p1x = [-18 -20 -18 -18 -20];    % x1-coordinates
p1y = [-18 -20 -20 -18 -18];    % x2-coordinates  
% p1y = p1x;
P1_composed = Polyhedron([p1x; p1y]');

% 2) Region where you lose package
% p2x = [-1 -1 2 2 -1]; % x1-coordinates
p2x = [-8 -8 -5 -5 -8];
p2y = [0 2 2 0 0];
% p2y = p2x;
P2_composed = Polyhedron([p2x; p2y]');

% 3) delivery region
p3x = [3 3 5 5 3]; % x1-coordinates
% p3y = [-.5 -2 -2 -.5 -.5]; % x2-coordinates
p3y = [5 3 3 5 5];
% p3y = p3x;
P3_composed = Polyhedron([p3x; p3y]');

% 4) strict avoid region 
p4x = [-8 -8 -5 -5 -8];
% p4x = [-7 -7 -5 -5 -7];
% p4x = [-10 -10 -4.5 -4.5 -10];
p4y = [-8 -5 -5 -8 -8];
% p4y = p4x;
P4_composed = Polyhedron([p4x; p4y]');



P1{1} = Polyhedron(combvec([min(p1x),max(p1x)])');
P1{2} = Polyhedron(combvec([min(p1y),max(p1y)])');
P2{1} = Polyhedron(combvec([min(p2x),max(p2x)])');
P2{2} = Polyhedron(combvec([min(p2y),max(p2y)])');
P3{1} = Polyhedron(combvec([min(p3x),max(p3x)])');
P3{2} = Polyhedron(combvec([min(p3y),max(p3y)])');
P4{1} = Polyhedron(combvec([min(p4x),max(p4x)])');
P4{2} = Polyhedron(combvec([min(p4y),max(p4y)])');


% regions that get specific atomic propositions
sys_region = cell(numel(A),1);
sys1_region = [P1{1}; P2{1};P3{1};P4{1}];
sys2_region = [P1{2}; P2{2};P3{2};P4{2}];
sys_region{1} = sys1_region;
sys_region{2} = sys2_region;


%% Synthesize scLTL formula
DFA.S = [1 2 3 4]; 
DFA.S0 = 2;
DFA.F = 1;
DFA.sink = [4];
DFA.act = {' ', 'p4', 'p3', 'p3p4', 'p2', 'p2p4', 'p2p3', 'p2p3p4', 'p1', 'p1p4', 'p1p3', 'p1p3p4', 'p1p2', ...
    'p1p2p4','p1p2p3', 'p1p2p3p4'};
% conservative for labeling
DFA.trans = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; ...
             2 4 2 4 2 4 2 4 3 4 3 4 3 4 3 4; ...
             3 4 1 4 2 4 2 4 3 4 1 4 2 4 1 4; ...
             4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4];

sysAP =  {'p1', 'p2', 'p3', 'p4'};


%% system abstraction
for i=1:numel(A)
    sysLTI{i} = LinModel(A(i), B(i), C(i), D(i), Bw(i), mu, sigma);
    sysLTI{i}.X = Polyhedron(combvec([xl,xu])');
    sysLTI{i}.U = Polyhedron(combvec([ul,uu])');
    sysLTI{i}.regions = sys_region{i};
    sysLTI{i}.AP = sysAP; 
    uhat{i} = GridInputSpace(lu,sysLTI{i}.U,'log');
    sysAbs{i} = tensorAbs_DFA(sysLTI{i},uhat{i},l,tol,DFA,'TensorComputation',true);
    sysAbs{i}.zstates = sysAbs{i}.states;
    %sysAbs{i} = FSabstraction(sysLTI{i},uhat{i},l,tol,DFA,'TensorComputation',true);
    P{i} = sysAbs{i}.P.Prob;
    nu = size(uhat{i},2);
    tensorP{i} = zeros(l,nu,l);
    for j = 1:nu
        submatrix = P{i}(:, (j-1)*l+1:j*l);
        tensorP{i}(:, j, :) = permute(submatrix, [1, 3, 2]);
    end
    % tensorP{i}= tensorP{i} ./ sum(tensorP{i}, 1);

end

TimeSysAbs = toc;

tic;

epsilon = 0.18; % larger than 0.3 is not advised.  

% Quantify similarity
simRel = cell(numel(A),1);
simRel{1} = QuantifySim_decoupled_2D(sysLTI{1}, sysAbs{1}, epsilon,DFA);
simRel{2} = QuantifySim_decoupled_2D(sysLTI{2}, sysAbs{2}, epsilon,DFA);

thold = 1e-6;

[satProb{1}] = SynthesizeRobustController_decoupled_2D(sysAbs{1}, DFA, simRel{1}, thold, tensorP{1}, true);
if isempty(nonzeros(satProb{1}))
    satProbtensor = zeros(l,l);
else
[satProb{2}] = SynthesizeRobustController_decoupled_2D(sysAbs{2}, DFA, simRel{2}, thold, tensorP{2}, true);


satProbtensor = outerProduct(satProb{1},satProb{2});
end
%% 
Time_VI = toc;

tic;
% Vvis = flipud(Vtensor);
Vvis = satProbtensor';
% Vvis =mat2gray(Vvis);
xLimits = [-20, 5];  % X-axis limits
yLimits = [-20, 5];  % Y-axis limits

% Calculate the number of grid points
numGridPointsX = l;
numGridPointsY = l;
x = linspace(xLimits(1), xLimits(2), numGridPointsX);
y = linspace(yLimits(1), yLimits(2), numGridPointsY);
imagesc(x, y, Vvis);
colorbar;

xlabel('X-axis');
ylabel('Y-axis');
set(gca, 'YDir', 'normal');  % Set Y-axis to normal direction
caxis([0, 1]);
hold on;
plot_x = plot(P1_composed);
set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
text(0.5*(min(p1x)+max(p1x))-1, 0.5*(min(p1y)+max(p1y)),'$P_1$','fontsize',15,'interpreter','latex');
hold on;
plot_x = plot(P2_composed);
set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
text(0.5*(min(p2x)+max(p2x))-1, 0.5*(min(p2y)+max(p2y)),'$P_2$','fontsize',15,'interpreter','latex');
hold on;
plot_x = plot(P3_composed);
set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
text(0.5*(min(p3x)+max(p3x))-1, 0.5*(min(p3y)+max(p3y)),'$P_3$','fontsize',15,'interpreter','latex');
hold on;
plot_x = plot(P4_composed);
% set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
set(plot_x, 'FaceColor','None','EdgeColor', 'w', 'LineWidth', 2.5); 
% text(0.5*(min(p4x)+max(p4x)), 0.5*(min(p4y)+max(p4y)),'$P_4$','fontsize',15,'interpreter','latex');
text(0.5*(min(p4x)+max(p4x))-1, 0.5*(min(p4y)+max(p4y)), '$P_4$', 'fontsize', 15, 'interpreter', 'latex', 'Color', 'w');

title('Visualization of the reachability probabilities iterated on the decoupled system');
TimeVisualization = toc;


d = whos();
d = sum([d.bytes]);
memory = d*10^(-6);

fprintf('Memory usage in total is: %.2f MB \n', memory);
fprintf('Time used for system definition and abstraction on the decoupled system is %.2f seconds\n', TimeSysAbs);
fprintf('Time used for value iteration on the decoupled system is %.2f seconds\n', Time_VI);
fprintf('Time used for visualization is %.2f seconds\n', TimeVisualization);



