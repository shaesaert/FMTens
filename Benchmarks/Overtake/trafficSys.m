function sys = trafficSys(deltaT)
%%% Function to create the LTI system for the scenario
%
% Input
% -----
% deltaT = the time between the time steps
% 
% Output
% -----
% sys = The LTI system modeling the traffic
%
    
    % Define state space matrices
    A = [1,0,deltaT,0;
         0,1,0,deltaT;
         0,0,1,0;
         0,0,0,1];
    B = [0,0;
         0,0;
         deltaT,0;
         0,deltaT];
    C = eye(4);
    D = zeros(4,2);
    Bw = eye(4);

    dim = length(A);

    % Specify mean and variance of disturbance w(t) 
    mu = zeros(dim,1); % mean of disturbance
    sigma = eye(dim); % variance of disturbance

    % Set up an LTI model   
    sys = LinModel(A,B,C,D,Bw,mu,sigma);
end