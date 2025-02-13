clc;
clear all;
close all;

%% given policy
% p: rank(policy) = 1;
% R_s: rank(indicator_safe) = 2; safe regions partitioned into 3 parts
% R_t: rank(indicator_target) = 1; 
% one reach-avoid problem, decomposed into 3 reach-avoid problems
% 2 dimensional system

tic;
%% system dynamics
dim = 2;
A = cell(dim,1);
B = cell(dim,1);
C = cell(dim,1);
D = cell(dim,1);
Bw = cell(dim,1);

mu = cell(dim,1);
sigma = cell(dim,1);

A{1} = 1;
A{2} = 1;
B{1} = 1;
B{2} = 1;
C{1} = 1;
C{2} = 1;
D{1} = 0;
D{2} = 0;
Bw{1} = 2*sqrt(0.25);
Bw{2} = 2*sqrt(0.25);
mu{1} = 0;
mu{2} = 0;
sigma{1} = 1;
sigma{2} = 1;

sysLTI = cell(dim,1);
uhat = cell(dim,1);
sysAbs = cell(dim,1);

%% state and input space
xl = -20;
xu = 5;

ul = -5;
uu = 5;

%% abstraction, 
% sysAbs{i}.Pxix: Probability(x,u,x+)
l = 100;
tol=10^-19;  
lu = 5; 

for i=1:dim
    sysLTI{i} = LinModel(A{i}, B{i}, C{i}, D{i}, Bw{i}, mu{i}, sigma{i});
    sysLTI{i}.X = Polyhedron(combvec([xl,xu])');
    sysLTI{i}.U = Polyhedron(combvec([ul,uu])');
    uhat{i} = GridInputSpace(lu,sysLTI{i}.U,'log');
    nu = size(uhat{i},2);
    sysAbs{i} = tensorAbs(sysLTI{i},uhat{i},l,tol,'TensorComputation',true);
end

%%
R = 20;
Rho = ones(l,1)/l;

%% 1st RA problem: define policy
part = 1;
Pol = cell(dim,1);
Pol{1} = zeros(l,nu);
Pol{2} = zeros(l,nu);

% Pol{1}(sysAbs{1}.states <= 0,5) = 1; % go up
Pol{1}(sysAbs{1}.states > 0,3) = 1; % stay

Pol{2}(sysAbs{2}.states <= 0,5) = 1; % go up
Pol{2}(sysAbs{2}.states > 0,3) = 1; % stay

%% define target and safe region for the 1st RA subproblem 
ind_target1 = zeros(l,1);
ind_target1(sysAbs{1}.states > 0) = 1;
ind_target2 = zeros(l,1);
ind_target2(sysAbs{2}.states > 0) = 1;

ind_safe1 = ind_target1;
ind_safe2 = zeros(l,1);
ind_safe2(sysAbs{2}.states <= 0) = 1;

%% mu is approximated min of value function that is transferred to 
% become the upper bound of NEXT  sub reach-avoid problem, mu is obtained
% using sampling
%% between each two sub reach-avoid problems, only mu is needed.
% Mfactor matrices are stored for visualization purpose and for final tensor
% value function intergration
%% Compute transition probabilities from state to state for each decoupled system

[mu,Mfactor_part1] = PI_subRA(R,l,Rho,part,Pol,ind_safe1,ind_safe2,ind_target1,ind_target2,sysAbs,dim);

figure;
tV_1 = Mfactor_part1{1}*Mfactor_part1{2}';
image((tV_1)*250)
title(['Overall Value function AFTER policy iteration over the ', num2str(part), ' th Reach-avoid subproblem']);
hold on;

%% pt2
part = 2;

%% define policy
Pol = cell(dim,1);
Pol{1} = zeros(l,nu);
Pol{2} = zeros(l,nu);

Pol{1}(sysAbs{1}.states <= 0 & sysAbs{1}.states > -8, 5) = 1; % up
Pol{1}(sysAbs{1}.states > 0,3) = 1; % stay

Pol{2}(:,3) = 1; % stay

%% define safe and target region
ind_target1 = ind_safe1*sqrt(mu);
ind_target2 = ind_safe2*sqrt(mu);

ind_safe1 = zeros(l,1);
ind_safe1(sysAbs{1}.states <= 0 & sysAbs{1}.states > -8 ) = 1;
ind_safe2 = zeros(l,1);
ind_safe2(sysAbs{2}.states <= 0) = 1;

[mu,Mfactor_part2] = PI_subRA(R,l,Rho,part,Pol,ind_safe1,ind_safe2,ind_target1,ind_target2,sysAbs,dim);

tV_2 = Mfactor_part2{1}*Mfactor_part2{2}';
mask = tV_1 ~= 0;
tV_2(mask) = tV_1(mask); % incorporate the value from the 1st subproblem

figure;
image((tV_2)*250)
title(['Overall Value function AFTER policy iteration over the', num2str(part), ' th Reach-avoid subproblem']);
hold on;

%% pt3
part = 3;

%% define policy
Pol = cell(dim,1);
Pol{1} = zeros(l,nu);
Pol{2} = zeros(l,nu);

Pol{1}(sysAbs{1}.states<= -8 ,5) = 1; % up


Pol{2}(sysAbs{2}.states <= 0,3) = 1; % stay
Pol{2}(sysAbs{2}.states > 0,1) = 1; % go down


%% define safe and target region
ind_target1 = ind_safe1*sqrt(mu);
ind_target2 = ind_safe2*sqrt(mu);


ind_safe1 = zeros(l,1);
ind_safe1(sysAbs{1}.states <= -8) = 1;
ind_safe2 = ones(l,1);


[mu,Mfactor_part3] = PI_subRA(R,l,Rho,part,Pol,ind_safe1,ind_safe2,ind_target1,ind_target2,sysAbs,dim);

tV_3 = Mfactor_part3{1}*Mfactor_part3{2}';
mask = tV_2 ~= 0;
tV_3(mask) = tV_2(mask); % incorporate the value from the previous 2 subproblem

figure;
image((tV_3)*250)
title(['Overall Value function AFTER policy iteration over the', num2str(part), ' th Reach-avoid subproblem']);
hold on;

TotalRuntime  = toc;

d = whos();
d = sum([d.bytes])
memory = d*10^(-6);

fprintf('Memory usage in total is: %.2f MB \n', memory);
fprintf('Total runtime is: %.2f seconds \n', TotalRuntime);
