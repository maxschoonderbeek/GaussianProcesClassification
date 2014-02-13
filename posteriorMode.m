function [post, logq] = posteriorMode(hyp, a, cov,lik,x,y)
% Algorithm 3.1 Rasmussen GPfML
% Find the posterior mode and approx log marginal likelihood
% Run IRLS Newton algorithm to optimise Psi(alpha).

%% Initialise variables
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
s = 0.6;                                        % Step-size in direction of da
a_new = a;

%% run Newton algorithm
while Psi_old - Psi_new > tol && it<maxit                       
    Psi_old = Psi_new; it = it+1;
    a_old = a_new;

    sW = sqrt(W); 
    L = chol(eye(n)+sW*sW'.*K);             % L'*L=B=eye(n)+sW*K*sW
    b = W.*f + dlp;                         % \
    B = sW.*(K*b);                          % | eq. (3.18) using eq. (3.27)
    a = b - sW.*(L\(L'\B));                 % /
    da = a - a_old;                         % Newton dir
    
    % Without line search:
    a_new = a_old + s*da;                   % Take smaller steps
    f = K*a_new;                            
    [lp,dlp,d2lp] = lik(y,f); W = -d2lp;    % Take W from the likelihood function 
    Psi_new = a_new'*f/2 - sum(lp);
    
    % With line search:
%     [s_line,Psi_new,n_line,dPsi_new,f,a_new,dlp,W] = search_line(a_old,da);
    
end 
it;
f = K*a;                                  % compute latent function values
[lp,dlp,d2lp] = lik(y,f); W = -d2lp; 
post.alpha = a;                            % return the posterior parameters
post.sW = sqrt(abs(W)).*sign(W);             % preserve sign in case of negative
post.L = chol(eye(n)+sW*sW'.*K);                   % recompute
post.f = f;                                     
logq = a'*f/2 + sum(log(diag(post.L))-lp);   % ..(f-m)/2 -lp +ln|B|/2

%% Psi function
% Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where 
% f = K*alpha Psi is the log-posterior
function [psi,dpsi,f,a,dlp,W] = Psi(a,K,lik,y)
  f = K*a;
  [lp,dlp,d2lp] = lik(y,f); W = -d2lp;
  psi = a'*f/2 - sum(lp);
  if nargout>1, dpsi = K*(a-dlp); end
  
  