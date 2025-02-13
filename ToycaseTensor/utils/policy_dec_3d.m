function pol = policy_dec_3d(Mfactor,Pxxi,l,Rho,R,d)

tV = zeros(l,l,l);
for i = 1:R
    tV = tV + outerProduct(Mfactor{1}(:,i),Mfactor{2}(:,i),Mfactor{3}(:,i));
end

if d == 1
    V_current_perm = permute(tV,[2,3,1]);


elseif d == 2
        V_current_perm = permute(tV,[1,3,2]);
elseif d ==3
    V_current_perm = tV;
end
    V_current = sum(V_current_perm .* Rho, [1,2]);
    V_current = reshape(V_current, [100, 1]);


    Pxi_x = reshape(Pxxi,[],l);
    Q = reshape(Pxi_x * V_current,l,[]);
    [~, I] = max(Q, [],2);
    pol = full(sparse(1:l,I,1));


end