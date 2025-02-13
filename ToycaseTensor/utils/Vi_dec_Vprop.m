% Value iteration
function Vi_tens_new = Vi_dec_Vprop(R,Vi_tens,Pxix, pol_i, ind_targeti,ind_safei)
%
% Pxxi [l*|a| x l]

% add policy  pol_i to Pxxi 

Pxxi = squeeze(sum((pol_i.* Pxix),2)); % shorter format

rank=size(Vi_tens,2);
% value iteration
for r = rank:-1:R+1
 Vi_tens_new(:, r) = ind_safei.*( Pxxi * Vi_tens(:, r-R));
end
 Vi_tens_new(:,1:R) = ind_targeti;


end



