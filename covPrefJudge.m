function K = covPrefJudge(hyp, x, y, i)
% Preference judgement kernel

% hyp = [ log(ell)
%         log(sf)  ]

if (nargin<3||numel(y)==0)
    y = x; 
end                                   % make sure y exists

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

k = @(x, y) sq_exp(x/ell, y/ell);
s = @(x, y) sq_dist(x/ell, y/ell);

Ktemp =sf2*(k(x(:,1), y(:,1)) + ...
            k(x(:,2), y(:,2)) - ...
            k(x(:,1), y(:,2)) - ...
            k(x(:,2), y(:,1)));

% Derivitives are used for hyperparemeter optimization:
if nargin<4                                                        % covariances
  K=Ktemp;
else                                            % derivatives - To be implemented
  if i==1
    K = sf2*(k(x(:,1), y(:,1))*s(x(:,1), y(:,1)) + ...
             k(x(:,2), y(:,2))*s(x(:,1), y(:,1)) - ...
             k(x(:,1), y(:,2))*s(x(:,1), y(:,1)) - ...
             k(x(:,2), y(:,1))*s(x(:,1), y(:,1)));
  elseif i==2
    K = 2*Ktemp;
  else
    error('Unknown hyperparameter')
  end
end


function C = sq_exp(x, y)
x=x';
y=y';
C = exp(-sq_dist(x, y)/2);

function C = sq_dist(a,b) % By substracting the mean the answer is numerrically more stable
    mu = 1/2*mean(b,2) + 1/2*mean(a,2);
    a = bsxfun(@minus,a,mu); b = bsxfun(@minus,b,mu);
    C = bsxfun(@plus,sum(a.*a,1)',bsxfun(@minus,sum(b.*b,1),2*a'*b));
    C = max(C,0);    % numerical noise can cause C to negative i.e. C > -1e-14