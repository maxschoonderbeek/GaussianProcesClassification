function lp = classPred(y, mu, s2)

kSigma = sqrt(1+pi.*s2/8);
lp = likLogistic1(y, kSigma.*mu);