function lp = predProbit(mu, s2, lik)
% Prediction of MacKay to soften the MAP prediction
kSigma = 1./sqrt(1+pi.*s2/8);
lp = lik([],kSigma.*mu);