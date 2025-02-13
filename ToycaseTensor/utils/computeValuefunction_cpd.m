function tV = computeValuefunction_cpd(Mfactor,R,varargin)

tV  = 0;
for k = 1:R
    tV = tV + outerProduct(Mfactor{1}(:,k),Mfactor{2}(:,k));
end


if nargin > 2
    Mfactor_overflow = varargin{1};
    tV = tV + outerProduct(Mfactor_overflow{1}(:,1),Mfactor_overflow{2}(:,1));


else
end


    tV = min(tV,1);
end