function pol = policy_dec(Mfactor,Pxxi, weights,l)

 
V_current = Mfactor*weights;
Pxi_x = reshape(Pxxi,[],l);

Q = reshape(Pxi_x * V_current,l,[]);
[~, I] = max(Q, [],2);
% pol = full(sparse(1:l,I,1));
pol = full(sparse(1:l,I,1,l,size(Q,2)));

end