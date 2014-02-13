function [lp,dlp,d2lp,d3lp] = likLogistic1(y, mu, s2)

% likLogistic - logistic function for binary classification or logit regression.
% The expression for the likelihood is 
%   likLogistic(t) = 1./(1+exp(-t)).
if numel(y)==0, y = ones(length(mu),1); end

f = mu; yf = y.*f; s = -yf;                   % product latents and labels
dlp = {}; d2lp = {}; d3lp = {};                         % return arguments
ps   = max(0,s); 
lp = -(ps+log(exp(-ps)+exp(s-ps)));                % lp = -(log(1+exp(s)))
if nargout>1                                           % first derivatives
  s   = min(0,f); 
  p   = exp(s)./(exp(s)+exp(s-f));                    % p = 1./(1+exp(-f))
  dlp = (y+1)/2-p;                          % derivative of log likelihood
  if nargout>2                          % 2nd derivative of log likelihood
    d2lp = -exp(2*s-f)./(exp(s)+exp(s-f)).^2;
    if nargout>3                        % 3rd derivative of log likelihood
      d3lp = 2*d2lp.*(0.5-p);
    end
  end
end


