function [lik] = invgampdf(x,a,b)
    lik = (b^a/gamma(a)) * x.^(-a-1) .* exp(-b./x);
end