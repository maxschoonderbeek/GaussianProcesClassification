function [post, logq] = posteriorMode(hyp, a, cov,lik,x,y)
% Algorithm 3.1
% Find the posterior mode and approx log marginal likelihood
% Run IRLS Newton algorithm to optimise Psi(alpha).

maxit = 30; 
Wmin = 0.0; 
tol = 1e-8; 
smin_line = 0; smax_line = 2;           % min/max line search steps size range
nmax_line = 10;                          % maximum number of line search steps
thr_line = 1e-4;                                       % line search threshold
nout = 5;

if isnumeric(cov),  K = cov;                    % use provided covariance matrix
else K = cov(hyp.cov,  x); end                  % evaluate the covariance matrix

Psi_line = @(s,a,da) Psi(a+s*da,K,lik,y);    % line search
pars_line = {smin_line,smax_line,nmax_line,thr_line};  % line seach parameters
search_line = @(a,da) brentmin(pars_line{:},Psi_line,nout,a,da);

f = K*a; 
[lp,dlp,d2lp] = lik(y,f); 
W = -d2lp; n = size(K,1);
Psi_new = Psi(a,K,lik,y);
Psi_old = Inf;  % make sure while loop starts by the largest old objective val
it = 0;                          % this happens for the Student's t likelihood
s = 0.5;                                        % Step-size in direction of da

while Psi_old - Psi_new > tol && it<maxit                       % begin Newton
    Psi_old = Psi_new; it = it+1;
    
    W = max(W,Wmin); % reduce step size by increasing curvature of problematic W
    sW = sqrt(W); L = chol(eye(n)+sW*sW'.*K);            % L'*L=B=eye(n)+sW*K*sW
    b = W.*f + dlp;
    B = sW.*(K*b);
    da = b - sW.*(L\(L'\B)) - a; % Newton dir + line search
    
    % Without line search:
    a = a + s*da;
    f = K*a;
    [lp,dlp,d2lp] = lik(y,f); W = -d2lp;
    Psi_new = a'*f/2 - sum(lp);
    
    % With line search:
%     [s_line,Psi_new,n_line,dPsi_new,f,a,dlp,W] = search_line(a,da);
    
end % end Newton's iterations
it;
f = K*a;                                  % compute latent function values
[lp,dlp,d2lp] = lik(y,f); W = -d2lp; 
post.alpha = a;                            % return the posterior parameters
post.sW = sqrt(abs(W)).*sign(W);             % preserve sign in case of negative
post.L = chol(eye(n)+sW*sW'.*K);                   % recompute
post.f = f;                                     
logq = a'*f/2 + sum(log(diag(post.L))-lp);   % ..(f-m)/2 -lp +ln|B|/2

% Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where 
% f = K*alpha+m, and likfun(f) = feval(lik{:},hyp.lik,y,  f,  [],inf).
% Psi is the log-posterior
function [psi,dpsi,f,a,dlp,W] = Psi(a,K,lik,y)
  f = K*a;
  [lp,dlp,d2lp] = lik(y,f); W = -d2lp;
  psi = a'*f/2 - sum(lp);
  if nargout>1, dpsi = K*(a-dlp); end
  
  