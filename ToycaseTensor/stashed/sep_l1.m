clc;
clear all;
close all;

%% given policy, 
% p: rank(policy) = 2,  
% reach-avoid, no DFA, 
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
Bw{1} = sqrt(0.25);
Bw{2} = sqrt(0.25);
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
% sysAbs{i}.Pxix stores the transition probability to the next state of (current state, current action) pair
l = 20; %% cannot be changed
tol=10^-3;  
lu = 5; 

for i=1:dim
    sysLTI{i} = LinModel(A{i}, B{i}, C{i}, D{i}, Bw{i}, mu{i}, sigma{i});
    sysLTI{i}.X = Polyhedron(combvec([xl,xu])');
    sysLTI{i}.U = Polyhedron(combvec([ul,uu])');
    uhat{i} = GridInputSpace(lu,sysLTI{i}.U,'log');
    nu = size(uhat{i},2);
    sysAbs{i} = tensorAbs(sysLTI{i},uhat{i},l,tol,'TensorComputation',true);
end

%% intialzie factor martices Mfactor which are CPD of value function, 
% set upper bound of rank of value function to be R
R = 10;
tol = 1; % iteration with SVD only implemented for a given policy
fprintf('Number of abstracted states on each dimension is %.f \n', l);
fprintf('Upper bound of rank of value function is %.f \n', R);
Mfactor = cell(dim,1);


%% define factor matrices to store CPD of value function (only implemented for 2D 
Mfactor{1} = zeros(l,R+tol);
Mfactor{2} = zeros(l,R+tol);

%% Initialize factor matrices for value iteration
% define safe and reach region
Mind_target = cell(dim,1);
Mind_safe = cell(dim,1);

%% target has no partition
Mind_target{1} = zeros(20,1);
Mind_target{2} = zeros(20,1);
% target partition 1/1
Mind_target{1}(1:10,1) = 1;
Mind_target{2}(10:20,1) = 1;


%% safe has 3 partitions
Mind_safe{1} = zeros(20,3);
Mind_safe{2} = zeros(20,3);

% safe partition 1/3
Mind_safe{1}(1:10,1) = 1;
Mind_safe{2}(1:9,1) = 1;

% safe partition 2/3
Mind_safe{1}(11:20,2) = 1;
Mind_safe{2}(1:9,2) = 1;

% safe partition 3/3
Mind_safe{1}(11:20,3) = 1;
Mind_safe{2}(10:20,3) = 1;

Mfactor{1}(:,size(Mind_target{1},2)) = Mind_target{1};
Mfactor{2}(:,size(Mind_target{2},2)) = Mind_target{2};

%% we can achieve the precise partition for safe and target regions
% via paritioning each dimension into 2
% i.e. each dimension has 2 partitions
par = 2;
%% design rank p policy
Pol_par1 = cell(dim,1);
Pol_par2 = cell(dim,1);

% partition 1 on dimension 1 and dimension 2
Pol_par1{1} = zeros(l,nu); % control state [1,10] to stay,no control for state [11,20]
Pol_par1{2} = zeros(l,nu); % control state [1,9] to go up,  no control for state [10,20]
Pol_par1{1}(1:10,3) = 1;
Pol_par1{2}(1:9,5) = 1;

% partition 2 on dimension 1 and dimension 2
Pol_par2{1} = zeros(l,nu); % control state [11,20] to go left/down, no control for state [1,10]
Pol_par2{2} = zeros(l,nu); % control state [10,20] to stay, no control for state [1,9]
Pol_par2{1}(11:20,1) = 1;
Pol_par2{2}(10:20,3) = 1;

% group (x,u) 
Pol_group = cell(dim,1);
Pol_group{1} = zeros(l*nu,par);
Pol_group{2} = zeros(l*nu,par);

% grouped policy on dimension 1
Pol_group{1}(:,1) = reshape(transpose(Pol_par1{1}),[l*nu,1]);
Pol_group{1}(:,2) = reshape(transpose(Pol_par2{1}),[l*nu,1]);

% grouped policy on dimension 2
Pol_group{2}(:,1) = reshape(transpose(Pol_par1{2}),[l*nu,1]);
Pol_group{2}(:,2) = reshape(transpose(Pol_par2{2}),[l*nu,1]);

%% one iteration
% % resize sysAbs.Pxix 
% P_re = cell(dim,1);
% for i = 1:l
%     P_re{1}(:,i) = reshape(transpose(sysAbs{1}.Pxix(:,:,i)),[l*nu,1]);
%     P_re{2}(:,i) = reshape(transpose(sysAbs{2}.Pxix(:,:,i)),[l*nu,1]);
% end

%% resize Pxx
Pxx1 = zeros(l, l);
for i = 1:l
    for j = 1:nu
        Pxx1(i, :) = Pxx1(i, :) + Pol_par1{1}(i, j) * squeeze(sysAbs{1}.Pxix(i, j, :))';
    end
end

Pxx2 = zeros(l, l);
for i = 1:l
    for j = 1:nu
        Pxx2(i, :) = Pxx2(i, :) + Pol_par2{1}(i, j) * squeeze(sysAbs{1}.Pxix(i, j, :))';
    end
end

%% update s =2,3,4,5,6,7
% z corresponds to safe set column
% l corresponds to policy column
% r corresponds to Mfactor column (from previous updated
% s corresponds to Mfactor column (to be updated this iteration

% s = 2 ,(r,l,z) = (1,1,1)
Mfactor{1}(:,2) = Mind_safe{1}(:,1) .* (Pxx1 * Mfactor{1}(:, 1));

% (1,2,1)
Mfactor{1}(:,3) = Mind_safe{1}(:,1) .* (Pxx2 * Mfactor{1}(:, 1));

% (1,1,2)
Mfactor{1}(:,4) = Mind_safe{1}(:,2) .* (Pxx1 * Mfactor{1}(:, 1));

% (1,2,2)
Mfactor{1}(:,5) = Mind_safe{1}(:,2) .* (Pxx2 * Mfactor{1}(:, 1));

% (1,1,3)
Mfactor{1}(:,6) = Mind_safe{1}(:,3) .* (Pxx1 * Mfactor{1}(:, 1));

% (1,2,3)
Mfactor{1}(:,7) = Mind_safe{1}(:,3) .* (Pxx2 * Mfactor{1}(:, 1));


%% dim 2
Pxx1 = zeros(l, l);
for i = 1:l
    for j = 1:nu
        Pxx1(i, :) = Pxx1(i, :) + Pol_par1{2}(i, j) * squeeze(sysAbs{2}.Pxix(i, j, :))';
    end
end

Pxx2 = zeros(l, l);
for i = 1:l
    for j = 1:nu
        Pxx2(i, :) = Pxx2(i, :) + Pol_par2{2}(i, j) * squeeze(sysAbs{2}.Pxix(i, j, :))';
    end
end
% s = 2 ,(r,l,z) = (1,1,1)
Mfactor{2}(:,2) = Mind_safe{2}(:,1) .* (Pxx1 * Mfactor{2}(:, 1));

% (1,2,1)
Mfactor{2}(:,3) = Mind_safe{2}(:,1) .* (Pxx2 * Mfactor{2}(:, 1));

% (1,1,2)
Mfactor{2}(:,4) = Mind_safe{2}(:,2) .* (Pxx1 * Mfactor{2}(:, 1));

% (1,2,2)
Mfactor{2}(:,5) = Mind_safe{2}(:,2) .* (Pxx2 * Mfactor{2}(:, 1));

% (1,1,3)
Mfactor{2}(:,6) = Mind_safe{2}(:,3) .* (Pxx1 * Mfactor{2}(:, 1));

% (1,2,3)
Mfactor{2}(:,7) = Mind_safe{2}(:,3) .* (Pxx2 * Mfactor{2}(:, 1));

tV = Mfactor{1} * Mfactor{2}';


TotalRuntime  = toc;

d = whos();
d = sum([d.bytes])
memory = d*10^(-6);

fprintf('Memory usage in total is: %.2f MB \n', memory);
fprintf('Total runtime is: %.2f seconds \n', TotalRuntime);
disp('Reachability probabilities of each state under the given policy is:')
disp(tV)


