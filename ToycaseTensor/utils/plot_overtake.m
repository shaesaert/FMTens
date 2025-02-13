function [] = plot_overtake(Mfactor,sysAbs, indices)
%PLOT_OVERTAKE Summary of this function goes here
%   Detailed explanation goes here
% Mfactor = cell with V_{grids,Rk} for each dimension
% sysAbs = cell with systems and info
n = size(sysAbs,1);

% do a reshape for each
for i =1:n
    Mfactor_re{i} =  reshape(Mfactor{i},[sysAbs{i}.l,size(Mfactor{i},2)]);
end


imagesc(sysAbs{1}.hx{1}, sysAbs{2}.hx{1},squeeze(Mfactor_re{1}(:,indices{1},:))*squeeze(Mfactor_re{2}(:,indices{2},:))')


end

