% Value iteration
function Vi_tens_new = Vi_dec(Vi_tens,Pxix, pol_i, ind_targeti,ind_safei)
%
% Pxxi [l*|a| x l]

% add policy  pol_i to Pxxi 

Pxxi = squeeze(sum((pol_i.* Pxix),2)); % shorter format

R=size(Vi_tens,2);
% value iteration
for r = R:-1:2
 Vi_tens_new(:, r) = ind_safei.*( Pxxi * Vi_tens(:, r-1));
end
 Vi_tens_new(:,1) = ind_targeti;


end



