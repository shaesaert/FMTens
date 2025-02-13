
clc
clear
close all

tic;
% Define system parameters
A = 0.9*eye(2);
B = 0.5*eye(2);
C = eye(2);
D = zeros(2);
Bw = sqrt(0.25)*eye(2);
dim = length(A);

% Specify mean and variance of disturbance w(t) 
mu = zeros(dim, 1); % Mean of disturbance
sigma = eye(dim); % Variance of disturbance

sysLTI = LinModel(A, B, C, D, Bw, mu, sigma);
 
% Bounds on state space 
x1l = -20; % Lowerbound in x1
x1u = 5; % Upperbound in x1
x2l = -20; % Lowerbound in x2
x2u = 5; % Upperbound in x2
sysLTI.X = Polyhedron(combvec([x1l, x1u], [x2l, x2u])');
TotalSpace = Polyhedron(combvec([x1l, x1u], [x2l, x2u])');

% Bounds on input space
ul = [-5; -5]; % Lowerbound input u
uu = [5; 5]; % Upperbound input u
sysLTI.U = Polyhedron(combvec([ul(1), uu(1)], [ul(2), uu(2)])');
% Specify regions for the specification
% Pick up a parcel at P1 and deliver it to P3. If on this path the agent passes 
% P2, it loses the package and has to pick up a new one (at P1).

% 1) Delivery region
p1x = [-18 -20 -18 -18 -20];    % x1-coordinates
p1y = [-18 -20 -20 -18 -18];    % x2-coordinates   
P1 = Polyhedron([p1x; p1y]');

% 2) Region where you lose package
p2x = [3 3 5 5 3]; % x1-coordinates
% p3y = [-.5 -2 -2 -.5 -.5]; % x2-coordinates
p2y = [5 3 3 5 5];
P2 = Polyhedron([p2x; p2y]');

% 3) Pick-up region
p3x = [-10 -10 -5 -5 -10];
p3y = [-8 -5 -5 -8 -8];
P3 = Polyhedron([p3x; p3y]');

% 4) strict avoid region 
% p4x = [-8 -8 -5 -5 -8];
% p4x = [-7 -7 -5 -5 -7];
p4x = [-3 -3 1.5 1.5 -3]; % x1-coordinates
p4y = [0 2 2 0 0];

P4 = Polyhedron([p4x; p4y]');

sysLTI.regions = [P1; P2; P3; P4]; % regions that get specific atomic propositions
sysLTI.AP = {'p1', 'p2', 'p3', 'p4'}; % with the corresponding atomic propositions

%Plot_sysLTI(sysLTI)

%% Step 1 Synthesize scLTL formula

DFA.S = [1 2 3 4]; 
DFA.S0 = 2;
DFA.F = 1;
DFA.sink = [4];
DFA.act = {' ', 'p4', 'p3', 'p3p4', 'p2', 'p2p4', 'p2p3', 'p2p3p4', 'p1', 'p1p4', 'p1p3', 'p1p3p4', 'p1p2', ...
    'p1p2p4','p1p2p3', 'p1p2p3p4'};
DFA.trans = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; ...
             2 4 2 4 2 4 2 4 3 4 3 4 3 4 3 4; ...
             3 4 1 4 2 4 2 4 3 4 1 4 2 4 1 4; ...
             4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4];


% Construct abstract input space uhat
lu = 14;  % number of abstract inputs in each direction
uhat = GridInputSpace(lu,sysLTI.U,'log'); 

% Construct finite-state abstraction
% l = [568,568];  % number of grid cells 
%l = [250,180]; %may7
%l = [500 412]; %may6
l = [568,563]; %may 10
% l = [510,470];  %may10 2
% l = [150 250];
tol=10^-6;  
sysAbs = FSabstraction(sysLTI,uhat,l,tol,DFA,'TensorComputation',true);

TimeSysAbs = toc;

%% Step 3 Similarity quantification


%[epsilonBounds] = ComputeEpsilonBounds(sysLTI,mu,sigma,sysAbs.beta)

% Choose a value for epsilon 
tic;

epsilon = 0.18; % larger than 0.3 is not advised.  

% Quantify similarity
simRel = QuantifySim(sysLTI, sysAbs, epsilon);

% t3end = toc(t3start);
%% Step 4 Synthesize a robust controller
% t4start = tic;

% Specify threshold for convergence error
thold = 1e-6;

% Synthesize an abstract robust controller
[satProb,pol] = SynthesizeRobustController(sysAbs, DFA, simRel, thold, true);

Time_VI = toc;
% t4end = toc(t4start);
%% Step 5 Control refinement

% Controller = RefineController(satProb,pol,sysAbs,simRel,sysLTI,DFA);

tic;
plotSatProb(satProb, sysAbs, 'initial', DFA);
set(gca, 'YDir', 'normal');  % Set Y-axis to normal direction
caxis([0, 1]);
% 
% 
% set(gca,'FontSize',26)
% set(gca,'TickLabelInterpreter','latex')
% 
% box on
% set(gca,'linewidth',1)

xlim([-20 5])
ylim([-20 5])

% enlarge fonts
set(gca,'FontSize',26)
set(gca,'TickLabelInterpreter','latex')
hold on;
plot_x = plot(P1);
set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
text(0.5*(min(p1x)+max(p1x))-1, 0.5*(min(p1y)+max(p1y)),'$P_1$','fontsize',15,'interpreter','latex');
hold on;
plot_x = plot(P2);
set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
text(0.5*(min(p2x)+max(p2x))-1, 0.5*(min(p2y)+max(p2y)),'$P_2$','fontsize',15,'interpreter','latex');
hold on;
plot_x = plot(P3);
set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
text(0.5*(min(p3x)+max(p3x))-1, 0.5*(min(p3y)+max(p3y)),'$P_3$','fontsize',15,'interpreter','latex');
hold on;
plot_x = plot(P4);
% set(plot_x, 'FaceColor','None', 'LineWidth', 2.5);
set(plot_x, 'FaceColor','None','EdgeColor', 'w', 'LineWidth', 2.5); 
% text(0.5*(min(p4x)+max(p4x)), 0.5*(min(p4y)+max(p4y)),'$P_4$','fontsize',15,'interpreter','latex');
text(0.5*(min(p4x)+max(p4x))-1, 0.5*(min(p4y)+max(p4y)), '$P_4$', 'fontsize', 15, 'interpreter', 'latex', 'Color', 'w');

TimeVisualization = toc;

d = whos();
d = sum([d.bytes]);
memory = d*10^(-6);

fprintf('Memory usage in total is: %.2f MB \n', memory);
fprintf('Time used for system definition and abstraction on the original system is %.2f seconds\n', TimeSysAbs);
fprintf('Time used for value iteration on the original system is %.2f seconds\n', Time_VI);
fprintf('Time used for visualization is %.2f seconds\n', TimeVisualization);


