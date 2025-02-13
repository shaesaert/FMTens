function Mfactor = PI_subRA_Vprop(R,nx,Rho,part,Pol,ind_safe1,ind_safe2,ind_target1,ind_target2,sysAbs,dim)

% if ~isempty(varargin) == 0
%     Mfactor_prev = varargin{1};
% else
%     Mfactor_prev = cell(dim,1);
%     Mfactor_prev{1} = zeros(l,R);
%     Mfactor_prev{2} = zeros(l,R);
% end

if ~iscell(Rho)
 rho = Rho;
 Rho = cell(dim,1);
 Rho{1} = rho;
 Rho{2} = rho;
else
end

if ~iscell(nx)
    l = nx;
    nx = cell(dim,1);
    nx{1} = l;
    nx{2} = l;
else
end

Pxx1 = squeeze(sum((Pol{1}.*sysAbs{1}.Pxix),2)); % shorter format
Pxx2 = squeeze(sum((Pol{2}.*sysAbs{2}.Pxix),2)); % shorter format

Mfactor = cell(dim,1);
R_sc = R;
R = size(ind_target1,2);
Mfactor{1} = zeros(nx{1},R_sc*R);
Mfactor{2} = zeros(nx{2},R_sc*R);
Mfactor{1}(:,1:R) = ind_target1;
Mfactor{2}(:,1:R) = ind_target2;

for r = R+1:R_sc*R
    Mfactor{1}(:, r) = ind_safe1.*( Pxx1 * Mfactor{1}(:, r-R));
    Mfactor{2}(:, r) = ind_safe2.*( Pxx2 * Mfactor{2}(:, r-R));
end

%% policy iteration
for index = 1:10

%% Given updated Value iterator, do 1 policy update
% estimate rho 1 and rho2 
% % Rho = ones(l,1)/l;
weights = Mfactor{2}'*Rho{1};
Pol{1} = policy_dec(Mfactor{1},sysAbs{1}.Pxix, weights,nx{1});

weights = Mfactor{1}'*Rho{2};
Pol{2} = policy_dec(Mfactor{2},sysAbs{2}.Pxix, weights,nx{2});

%% Given updated policy, do 1 value update 
Mfactor{1}  = Vi_dec_Vprop(R,Mfactor{1},sysAbs{1}.Pxix, Pol{1},ind_target1,ind_safe1);
Mfactor{2} = Vi_dec_Vprop(R,Mfactor{2},sysAbs{2}.Pxix, Pol{2},ind_target2,ind_safe2);

end


end
