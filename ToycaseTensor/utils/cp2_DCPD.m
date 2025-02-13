function [A, B, lam, Xhat] = cp2_DCPD(X, R)
    % Number of iterations
    Iter1 = 50;
    Iter2 = 3;

    % Initialize factor matrices with random values
    A = randn(size(X, 1), R);
    B = randn(size(X, 2), R);

    % Outer iteration loop
    for i = 1:Iter1
        for r = 1:R
            % Exclude current rank r
            ind = setdiff(1:R, r);

            % Compute weights for the components excluding the current rank
            lam = lam_find(X, A(:, ind), B(:, ind));

            % Reconstruct the matrix excluding the current rank
            EST = matrix_reconst(A(:, ind), B(:, ind), lam);

            % Inner iteration loop
            for i2 = 1:Iter2
                % Update factor matrices
                A(:, r) = (X - EST) * B(:, r) / (B(:, r)' * B(:, r));
                B(:, r) = (X - EST)' * A(:, r) / (A(:, r)' * A(:, r));
            end
        end
    end

    % Compute final weights
    lam = lam_find(X, A, B);

    % Reconstruct the approximate matrix
    Xhat = matrix_reconst(A, B, lam);
end

%%%%%% FUNCTIONS %%%%%%%%%

function Xhat = matrix_reconst(A, B, lam)
    % Reconstruct matrix from factor matrices and weights
    Xhat = (A * diag(lam)) * B';
end

function lam = lam_find(X, A, B)
    % Compute weights for the given matrix and factor matrices
    for r = 1:size(A, 2)
        D(:, r) = vec(matrix_reconst(A(:, r), B(:, r), 1));
    end
    lam = pinv(D) * X(:);
end

function v = vec(x)
    % Vectorize the input matrix
    v = reshape(x, numel(x), 1);
end
