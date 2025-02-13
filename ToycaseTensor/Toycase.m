function [sysLTI,sysAbs, uhat] = Toycase(l, lu)
%TOYCASE Summary of this function goes here
%   Detailed explanation goes here
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
tol=10^-19;  

for i=1:dim
    sysLTI{i} = LinModel(A{i}, B{i}, C{i}, D{i}, Bw{i}, mu{i}, sigma{i});
    sysLTI{i}.X = Polyhedron(combvec([xl,xu])');
    sysLTI{i}.U = Polyhedron(combvec([ul,uu])');
    uhat{i} = GridInputSpace(lu,sysLTI{i}.U,'log');
    nu = size(uhat{i},2);
    sysAbs{i} = tensorAbs(sysLTI{i},uhat{i},l,tol,'TensorComputation',true);
end
end

