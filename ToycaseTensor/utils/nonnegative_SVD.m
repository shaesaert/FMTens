function [U,V] = nonnegative_SVD(U,V)
for i = 1:size(V, 2)
    if any(U(:, i) < 0)
        U(:, i) = -U(:, i);
        V(:, i) = -V(:, i);
    end
end
end