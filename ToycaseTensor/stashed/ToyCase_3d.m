clc;
clear all;
close all;

%% 
% given policy, 
% p: rank(policy) = 1, 
% reach-avoid, no DFA, 
% 3 dimensional system

tic;
%% system dynamics
dim = 3;
A = cell(dim,1);
B = cell(dim,1);
C = cell(dim,1);
D = cell(dim,1);
Bw = cell(dim,1);

mu = cell(dim,1);
sigma = cell(dim,1);

A{1} = 1;
A{2} = 1;
A{3} = 1;
B{1} = 1;
B{2} = 1;
B{3} = 1;
C{1} = 1;
C{2} = 1;
C{3} = 1;
D{1} = 0;
D{2} = 0;
D{3} = 0;
Bw{1} = 2*sqrt(0.25);
Bw{2} = 2*sqrt(0.25);
Bw{3} = 2*sqrt(0.25);
mu{1} = 0;
mu{2} = 0;
mu{3} = 0;
sigma{1} = 1;
sigma{2} = 1;
sigma{3} = 1;

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


%% Define (given) policies: Pol{1}, Pol{2}
Pol = cell(dim,1);
Pol{1} = zeros(l,nu);
Pol{1}(:,3) = 1; % stay 

Pol{2} = zeros(l,nu);
Pol{2}(sysAbs{2}.states<-5,5) = 1; % go up 
Pol{2}(sysAbs{2}.states>=-5&sysAbs{2}.states<-4,4) = 1; % go up 
Pol{2}(sysAbs{2}.states>=-4&sysAbs{2}.states<-3,3) = 1; % stay
Pol{2}(sysAbs{2}.states>=-3&sysAbs{2}.states<-1,2) =1; % go down 
Pol{2}(sysAbs{2}.states>=-1,1) = 1; % go down

Pol{3} = zeros(l,nu);
Pol{3}(:,3) = 1; % stay 

%% Compute transition probabilities from state to state for each decoupled system
Pxx1 = squeeze(sum((Pol{1}.*sysAbs{1}.Pxix),2)); % shorter format
Pxx2 = squeeze(sum((Pol{2}.*sysAbs{2}.Pxix),2)); % shorter format
Pxx3 = squeeze(sum((Pol{3}.*sysAbs{3}.Pxix),2)); % shorter format
% for i = 1:l
%     for j = 1:nu
%         Pxx1(i, :) = Pxx1(i, :) + Pol{1}(i, j) * squeeze(sysAbs{1}.Pxix(i, j, :))';
%         Pxx2(i, :) = Pxx2(i, :) + Pol{2}(i, j) * squeeze(sysAbs{2}.Pxix(i, j, :))';
%     end
% end

%% Initialize factor matrices for value iteration
% define safe and reach region
ind_target1 = ones(l,1);
ind_target2 = zeros(l,1);
ind_safe1 = ones(l,1);
ind_safe2 = ones(l,1);
% target [-4,-2]
ind_target2(sysAbs{2}.states<-2 & sysAbs{2}.states>-4) = 1;
ind_safe2(sysAbs{2}.states<-2 & sysAbs{2}.states>-4) = 0;

ind_target3 = ones(l,1);
ind_safe3  = ones(l,1);

%% intialize factor martices Mfactor which are CPD of value function, 
R = 20; % set upper bound of rank of value function to be R
fprintf('Number of abstracted states on each dimension is %.f \n', l);
fprintf('Upper bound of rank of value function is %.f \n', R);
Mfactor = cell(dim,1);


%% define factor matrices to store CPD of value function V^(1) V^(2)
Mfactor{1} = zeros(l,R);
Mfactor{2} = zeros(l,R);
Mfactor{3} = zeros(l,R);


Mfactor{1}(:,1) = ind_target1;
Mfactor{2}(:,1) = ind_target2;
Mfactor{3}(:,1) = ind_target3;


%% Iteration of R-1 steps
%% TODO: encode ind_safe in Pxx for p = 1
%% Compute value iteration for 2-R
for r = 2:R
    Mfactor{1}(:, r) = ind_safe1.*( Pxx1 * Mfactor{1}(:, r-1));
    Mfactor{2}(:, r) = ind_safe2.*( Pxx2 * Mfactor{2}(:, r-1));
    Mfactor{3}(:, r) = ind_safe3.*( Pxx3 * Mfactor{3}(:, r-1));
end

tV = zeros(l,l,l);
for i = 1:R
    tV = tV + outerProduct(Mfactor{1}(:,i),Mfactor{2}(:,i),Mfactor{3}(:,i));
end


for index = 1:10

%% Given updated Value iterator, do 1 policy update
% estimate rho 1 and rho2 
Rho = ones(l,l)/(l^2);

Pol{1} = policy_dec_3d(Mfactor,sysAbs{1}.Pxix, l,Rho,R,1);
Pol{2} = policy_dec_3d(Mfactor,sysAbs{2}.Pxix, l,Rho,R,2);
Pol{3} = policy_dec_3d(Mfactor,sysAbs{3}.Pxix, l,Rho,R,3);


%% Given updated policy, do 1 value update 
Mfactor{1}  = Vi_dec(Mfactor{1},sysAbs{1}.Pxix, Pol{1},ind_target1,ind_safe1);
Mfactor{2} = Vi_dec(Mfactor{2},sysAbs{2}.Pxix, Pol{2},ind_target2,ind_safe2);
Mfactor{3} = Vi_dec(Mfactor{3},sysAbs{3}.Pxix, Pol{3},ind_target3,ind_safe3);




tV = zeros(l,l,l);
for i = 1:R
    tV = tV + outerProduct(Mfactor{1}(:,i),Mfactor{2}(:,i),Mfactor{3}(:,i));
end

title('during iteration')
disp(tV)
index  = index+1;
% pause

end
title('after iteration')

TotalRuntime  = toc;

d = whos();
d = sum([d.bytes])
memory = d*10^(-6);

fprintf('Memory usage in total is: %.2f MB \n', memory);
fprintf('Total runtime is: %.2f seconds \n', TotalRuntime);
disp('Reachability probabilities of each state under the given policy is:')
disp(tV)


