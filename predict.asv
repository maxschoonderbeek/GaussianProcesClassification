function [ymu, ys2, fmu, fs2] = predict(hyp, post, cov, lik, piPred, x, xs)
% Algorithm 3.2 Rasmussen GPfML

%% Initialize some variables
alpha = post.alpha; L = post.L; sW = post.sW;
ns = size(xs,1);                                       % number of data points
ymu = zeros(ns,1); ys2 = ymu; fmu = ymu; fs2 = ymu; lp = ymu;   % allocate mem
nz = true(size(alpha,1),1);

%% The algorithm is given as follows 
% ks = cov(hyp.cov, x, xs);
% kss = ones(ns,1)*exp(2*hyp.cov(2));     % Self-variance
% fmu = ks'*full(alpha);                  % Latent mean (3.21)
% v = L'\(diag(sW)'*ks);                  % \ Latent variances
% fs2 = kss-diag(v'*v);                   % / (3.24)/(3.29)                    
% [Lp] = piPred(fmu(:),fs2,lik);              % \ 
% p = exp(Lp);                            % | Predictive class probability
% ymu = 2*p-1;                            % | Class mean
% ys2 = 4*p.*(1-p);                       % / Class variance

%% To handle large batches we use the following algorithm
% This is in principle the same as the algorithm above, it only handels
% smaller batches of the prediction data
nperbatch = 1000;               % number of data points per mini batch
nact = 0;                       % number of already processed test data points
while nact<ns                   % process minibatches of test cases to save memory
    id = (nact+1):min(nact+nperbatch,ns);               % data points to process 
    Ks  = cov(hyp.cov,  x(nz,:), xs(id,:));             % cross-covariances
    kss = ones(id(end)-id(1)+1,1)*exp(2*hyp.cov(2));    % self-variance                     
    Fmu = Ks'*full(alpha(nz,:));                        % conditional mean fs|f
    fmu(id) = sum(Fmu,2);                               % predictive means
    V  = L'\(repmat(sW,1,length(id)).*Ks);                      
    fs2(id) = kss - sum(V.*V,1)'; % predictive variances (3.24)/(3.29)
    fs2(id) = max(fs2(id),0);   % remove numerical noise i.e. negative variances
    [Lp] = piPred(Fmu(:),fs2(id),lik);
    p = exp(Lp);
    Ymu = 2*p-1;                % first y moment
    Ys2 = 4*p.*(1-p);           % second y moment
    lp(id)  = Lp;               % log probability; sample averaging
    ymu(id) = Ymu;              % predictive mean ys|y and ..
    ys2(id) = Ys2;              % .. variance    
    nact = id(end);             % set counter to index of last processed data point
end


