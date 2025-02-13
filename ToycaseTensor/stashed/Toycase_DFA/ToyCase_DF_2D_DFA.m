%% fig.4b
clc; clear; close all;

% Add toolboxes to path
run Install.m

% Track runtime
tStart = tic;
disp('Start package delivery benchmark')

%% Specify system and regions
% LTI system of the form
% 
% x(t+1) = A x(t) + B u(t) + Bw w(t) 
% y(t) = C x(t) + D u(t)

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
p2x = [0 0 2 2 0]; % x1-coordinates
p2y = [0 2 2 0 0];
P2 = Polyhedron([p2x; p2y]');

% 3) Pick-up region
p3x = [3 3 5 5 3]; % x1-coordinates
% p3y = [-.5 -2 -2 -.5 -.5]; % x2-coordinates
p3y = [5 3 3 5 5];
P3 = Polyhedron([p3x; p3y]');

% 4) strict avoid region 
p4x = [-8 -8 -5 -5 -8];
% p4y = [-4 0 0 -4 -4];
p4y = [-8 -5 -5 -8 -8];
P4 = Polyhedron([p4x; p4y]');


sysLTI.regions = [P1; P2; P3; P4]; % regions that get specific atomic propositions
sysLTI.AP = {'p1', 'p2', 'p3', 'p4'}; % with the corresponding atomic propositions

%Plot_sysLTI(sysLTI)

%% Step 1 Synthesize scLTL formula

%formula = 'F(p1 & (!p2 U p3))';
%[DFA] = TranslateSpec(formula,sysLTI.AP);

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


delta = 0.5;
epsilonDF = compute_epsilonDF(delta,A,B,Bw);
rel = SimRel(epsilonDF,delta,D);
% simulation relation of PRM model

% Inflate polytopes of interested regions and state spaces for labelling
P = sysLTI.regions;
[PolytopeLargeSS,PolytopeSmallSS,LargePolytopes,SmallPolytopes] = bi_inflation_polytopes(sysLTI,P,epsilonDF);

%% step 2.2. update PRM model via RRG1
nrOfSteps = 3; % fixed for current implementation
K_new = 48; 
K_new_fb = 3; 
rng("default")
s=rng;

%% RUOHAN-TODO: sampling
%sysAbs_PRM = PRM_hp_aug(sysLTI,P,nrOfSteps,DFA); % to generate workspace hp_error2
% rng(1,"philox")
sysAbs_PRM = PRM_model_flex_pureRRG(K_new,K_new_fb,sysLTI,PolytopeSmallSS,LargePolytopes,SmallPolytopes,P,nrOfSteps,DFA,epsilonDF);
% TODO: reduce number of inputs of this type of functions.for example x1u,
% x1l,... all part of the sysLTI properties. Also coding like this is not
% scalable. 
sysAbs_PRM = DeterministicLabelling_PRM(sysAbs_PRM, DFA, sysAbs_PRM.regions, sysAbs_PRM.APs);
NonDetLabels = NonDeterministicLabelling_PRM(sysAbs_PRM.outputs, sysLTI.regions, rel, 'Efficient',sysAbs_PRM);
rel.NonDetLabels = NonDetLabels;
rel_PRM = rel;
% NonDetLabels=[1 1 0 1 0 1;
%               0 0 1 0 0 0;
%               0 0 0 0 1 0;
%               0 0 0 0 0 0;];
% rel.NonDetLabels = NonDetLabels;
% rel_PRM = rel;

%plot(sysAbs_PRM.Graph, 'XData', sysAbs_PRM.states(1,:), 'YData', sysAbs_PRM.states(2,:));

thold = 1e-6;     % threshold
satProb= PureRRGDP(sysAbs_PRM, rel_PRM, DFA,delta,sysLTI);
hold on;   
plotSatProb_MM_pureRRG(epsilonDF,satProb,sysAbs_PRM, sysLTI, 'initial', DFA)
% plotSatProb_pureRRG(satProb,sysAbs_PRM);
% figure;
% Plot_sysLTI(sysLTI);

tEnd = toc(tStart);
disp(['Total runtime = ', mat2str(tEnd,3), ' seconds']) 

if sysAbs_PRM.num_ssc == 1
    fprintf('Graph of PRM model is strongly-connected');
else
    fprintf('Graph of PRM model is not strongly-connected but the number of strongly-connected components of it is %d.\n', sysAbs_PRM.num_ssc);
end


set(gca,'FontSize',26)
set(gca,'TickLabelInterpreter','latex')

box on
set(gca,'linewidth',1)

xlim([-20 5])
ylim([-20 5])

% enlarge fonts
set(gca,'FontSize',26)
set(gca,'TickLabelInterpreter','latex')

