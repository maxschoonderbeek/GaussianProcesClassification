function [post,nlZ,dnlZ] = inferLaplace(hyp, cov, lik, x, y)

% Laplace approximation to the posterior Gaussian process.
n = size(x,1);
persistent last_alpha                                   % copy of the last alpha
if any(isnan(last_alpha)), last_alpha = zeros(size(last_alpha)); end   % prevent
if any(size(last_alpha)~=[n,1]), alpha = zeros(n,1);      % set alpha first time
else  alpha = last_alpha; end

if isnumeric(cov),  K = cov;                    % use provided covariance matrix
else K = cov(hyp.cov,  x); end                  % evaluate the covariance matrix

likfun = @(f) lik(y,f);

[post,nlZ] = posteriorMode(hyp,alpha,K,lik,x,y); 


alpha = post.alpha; sW = post.sW; L = post.L; f = post.f;
[~,dlp,~,d3lp] = likfun(f);

last_alpha = alpha;                                     % remember for next call

%% Algorithm 5.1
if nargout>2                                           % do we want derivatives?
  dnlZ = hyp;                                   % allocate space for derivatives
  R = repmat(sW,1,n).* (L\(L'\diag(sW)));     %sW*inv(B)*sW=inv(K+inv(W))
  C = post.L'\(repmat(sW,1,n).*K);                     % deriv. of ln|B| wrt W
  g = (diag(K)-sum(C.^2,1)')/2;                    % g = diag(inv(inv(K)+W))/2
  dfhat = g.*d3lp;  % deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
  for i=1:length(hyp.cov)                                    % covariance hypers
    dK = cov(hyp.cov, x, [], i);
    s1 = sum(sum(R.*dK))/2 - alpha'*dK*alpha/2;         % explicit part
    b = dK*dlp;                            % b-K*(Z*b) = inv(eye(n)+K*diag(W))*b
    s3 = b-K*(R*b);
    dnlZ.cov(i) = s1 - dfhat'*( s3 );            % implicit part
  end
end

  