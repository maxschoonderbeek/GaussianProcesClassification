function [ymu, ys2, fmu, fs2] = predict(hyp, post, cov, lik, x, xs)
% Algorithm 3.2
alpha = post.alpha; L = post.L; sW = post.sW;
ns = size(xs,1);                                       % number of data points
ymu = zeros(ns,1); ys2 = ymu; fmu = ymu; fs2 = ymu; lp = ymu;   % allocate mem
nz = true(size(alpha,1),1);

% % The algorithm is given as follows 
% ks = cov(hyp.cov, x, xs);
% % kss = cov(hyp.cov, xs, xs); % We need a trick to prevent that matlab runs out of memory
% fmu = ks'*full(alpha);
% v = L\(diag(sW)'*ks);
% kss = ones(ns,1)*exp(2*hyp.cov(2));
% fs2 = kss-diag(v'*v);                         % variance as in MLaPP 15.3            

% To handle large batches we use the following algorithm
nperbatch = 1000;                       % number of data points per mini batch
nperbatch = ns;
nact = 0;                       % number of already processed test data points
while nact<ns               % process minibatches of test cases to save memory
    id = (nact+1):min(nact+nperbatch,ns);               % data points to process 
    Ks  = cov(hyp.cov,  x(nz,:), xs(id,:));             % cross-covariances
    kss = ones(id(end)-id(1)+1,1)*exp(2*hyp.cov(2));    % self-variance                     
    Fmu = Ks'*full(alpha(nz,:));                         % conditional mean fs|f
    fmu(id) = sum(Fmu,2);                                   % predictive means
    V  = L'\(repmat(sW,1,length(id)).*Ks);                      
    fs2(id) = kss - sum(V.*V,1)';           % predictive variances (3.24)/(3.29)
    fs2(id) = max(fs2(id),0);   % remove numerical noise i.e. negative variances
    y = ones(size(Fmu(:)));
    [Lp] = lik(y,Fmu(:),fs2(id));
    p = exp(Lp);
    Ymu = 2*p-1;                                                % first y moment
    Ys2 = 4*p.*(1-p);                                          % second y moment
    lp(id)  = Lp;                            % log probability; sample averaging
    ymu(id) = Ymu;                                 % predictive mean ys|y and ..
    ys2(id) = Ys2;                                                 % .. variance    
    nact = id(end);          % set counter to index of last processed data point
end


