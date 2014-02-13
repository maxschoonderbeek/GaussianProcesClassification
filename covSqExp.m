function K = covSqExp(hyp, x, z, i)
% Squared Exponential covariance function with isotropic distance measure. The
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
%
% where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
% variance. The hyperparameters are:
%
% hyp = [ log(ell)
%         log(sf)  ]

if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; 

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

% precompute squared distances
if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist1(x'/ell,x'/ell);
else                                                   % cross covariances Kxz
    K = sq_dist1(x'/ell,z'/ell);
end

if nargin<4                                                        % covariances
  K = sf2*exp(-K/2);
else                                                               % derivatives
  if i==1
    K = sf2*exp(-K/2).*K;
  elseif i==2
    K = 2*sf2*exp(-K/2);
  else
    error('Unknown hyperparameter')
  end
end


function C = sq_dist1(a,b) % By substracting the mean the answer is numerrically more stable
    mu = 1/2*mean(b,2) + 1/2*mean(a,2);
    a = bsxfun(@minus,a,mu); b = bsxfun(@minus,b,mu);
    C = bsxfun(@plus,sum(a.*a,1)',bsxfun(@minus,sum(b.*b,1),2*a'*b));
    C = max(C,0);    % numerical noise can cause C to negative i.e. C > -1e-14

