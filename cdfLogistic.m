function [lp,ymu,ys2] = cdfLogistic(mu, s2,lik)

% likLogistic - logistic function for binary classification or logit regression.
% The expression for the likelihood is 
%   likLogistic(t) = 1./(1+exp(-t)).
%
% The moments
% \int f^k likLogistic(y,f) N(f|mu,var) df are calculated via a cumulative 
% Gaussian scale mixture approximation.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-09-02.
%
% See also LIKFUNCTIONS.M.

inf = 'infEP';


y = ones(size(mu));                                       % make y a vector

% likLogistic(t) \approx 1/2 + \sum_{i=1}^5 (c_i/2) erf(lam_i/sqrt(2)t)
lam = sqrt(2)*[0.44 0.41 0.40 0.39 0.36];    % approx coeffs lam_i and c_i
c = [1.146480988574439e+02; -1.508871030070582e+03; 2.676085036831241e+03;
  -1.356294962039222e+03;  7.543285642111850e+01                      ];
% lZc = likErf([], y*ones(1,5), mu*lam, s2*(lam.^2), inf);
z = mu*lam./sqrt(1+s2*(lam.^2));
lZc = logphi(z);

lp = log_expA_x(lZc,c);       % A=lZc, B=dlZc, d=c.*lam', lZ=log(exp(A)*c)

% The scale mixture approximation does not capture the correct asymptotic
% behavior; we have linear decay instead of quadratic decay as suggested
% by the scale mixture approximation. By observing that for large values 
% of -f*y ln(p(y|f)) for likLogistic is linear in f with slope y, we are
% able to analytically integrate the tail region.
val = abs(mu)-196/200*s2-4;       % empirically determined bound at val==0
lam = 1./(1+exp(-10*val));                         % interpolation weights
lZtail = min(s2/2-abs(mu),-0.1);  % apply the same to p(y|f) = 1 - p(-y|f)
id = y.*mu>0; lZtail(id) = log(1-exp(lZtail(id)));  % label and mean agree
lp   = (1-lam).*  lp + lam.*  lZtail;      % interpolate between scale ..

ymu = {}; ys2 = {};
if nargout>1
    p = exp(lp);
    ymu = 2*p-1;                                                % first y moment
    if nargout>2
      ys2 = 4*p.*(1-p);                                        % second y moment
    end
end

%  computes y = log( exp(A)*x ) in a numerically safe way by subtracting the
%  maximal value in each row to avoid cancelation after taking the exp
function y = log_expA_x(A,x)
  N = size(A,2);  maxA = max(A,[],2);      % number of columns, max over columns
  y = log(exp(A-maxA*ones(1,N))*x) + maxA;  % exp(A) = exp(A-max(A))*exp(max(A))
  
%  computes y = ( (exp(A).*B)*z ) ./ ( exp(A)*x ) in a numerically safe way
%  The function is not general in the sense that it yields correct values for
%  all types of inputs. We assume that the values are close together.
function y = expABz_expAx(A,x,B,z)
  N = size(A,2);  maxA = max(A,[],2);      % number of columns, max over columns
  A = A-maxA*ones(1,N);                                 % subtract maximum value
  y = ( (exp(A).*B)*z ) ./ ( exp(A)*x );
