function mu_approx = sample_mu(Mfactor,noSample,l)
 sample_mu = inf;

for k = 1:noSample
    i = randi([1 l]);
    j = randi([1 l]);
    dp = Mfactor{1}(i,:) * Mfactor{2}(j,:)';
    if dp ~= 0 && dp< sample_mu
        sample_mu = dp;
    end
end

mu_approx = sample_mu;
end