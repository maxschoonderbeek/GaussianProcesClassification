function lp = predErf(mu,s2,lik)
% Class prediction as in barber BRaML
erflambda = sqrt(pi)/4;
lp = log(0.5 + 0.5*erf(erflambda * mu./sqrt(1 + 2*erflambda^2 * s2)));
