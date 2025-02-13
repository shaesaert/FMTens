function [mu,tV_end] = PI(l,nu,sysAbs,R,dim,Rho,Pol,ind_target1,ind_target2,ind_safe1,ind_safe2,part,varargin)
if ~isempty(varargin)
    tV_last = varargin{1};
    
else
    tV_last = zeros(l,l);
end

%% Compute transition probabilities from state to state for each decoupled system
Pxx1 = squeeze(sum((Pol{1}.*sysAbs{1}.Pxix),2)); % shorter format
Pxx2 = squeeze(sum((Pol{2}.*sysAbs{2}.Pxix),2)); % shorter format


%% intialize factor martices Mfactor which are CPD of value function, 
Mfactor = cell(dim,1);

Mfactor{1} = zeros(l,R);
Mfactor{2} = zeros(l,R);
Mfactor{1}(:,1) = ind_target1;
Mfactor{2}(:,1) = ind_target2;

for r = 2:R
    Mfactor{1}(:, r) = ind_safe1.*( Pxx1 * Mfactor{1}(:, r-1));
    Mfactor{2}(:, r) = ind_safe2.*( Pxx2 * Mfactor{2}(:, r-1));
end




tV_b = max(Mfactor{1} * Mfactor{2}',tV_last);
figure;
image(tV_b*250)
title(['Value function BEFORE policy iteration over ', num2str(part), ' part of decomposed DFA']);


% figure
for index = 1:10

%% Given updated Value iterator, do 1 policy update
% estimate rho 1 and rho2 
% % Rho = ones(l,1)/l;
weights = Mfactor{2}'*Rho;
Pol{1} = policy_dec(Mfactor{1},sysAbs{1}.Pxix, weights,l);

weights = Mfactor{1}'*Rho;
Pol{2} = policy_dec(Mfactor{2},sysAbs{2}.Pxix, weights,l);

%% Given updated policy, do 1 value update 
Mfactor{1}  = Vi_dec(Mfactor{1},sysAbs{1}.Pxix, Pol{1},ind_target1,ind_safe1);
Mfactor{2} = Vi_dec(Mfactor{2},sysAbs{2}.Pxix, Pol{2},ind_target2,ind_safe2);

end
% tV_af = Mfactor{1} * Mfactor{2}';
% mu = min(nonzeros(tV_af));
mu_approx = sample_mu(Mfactor,0.3*R,l);
mu = mu_approx;


tV_af = max(Mfactor{1} * Mfactor{2}',tV_b);
figure;
image(tV_af*250)
title(['Value function AFTER policy iteration over ', num2str(part), ' part of decomposed DFA']);


tV_end = tV_af;
end