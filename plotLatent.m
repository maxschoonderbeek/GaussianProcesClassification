function plotLatent(mu2D,xx,V)

figure; hold all;

m = length(V);
ntr = size(mu2D,1);
mu_sel = zeros(ntr,m);

yN = sum(mu2D)/ntr;
yN = -yN;
plot(xx, yN,'r');
leg = {'Averaged function'};
[yMx, loc1] = max(yN);
plot(xx(loc1),yMx,'bo')
leg{end+1} = sprintf('Maximum');

% plot(xx, mu2D)

for n = 1:m
    mu_sel(:,n) = mu2D(:,V(n));
%     mmin = min(yNormx(:,n));
%     mmax = max(yNormx(:,n));
    plot(xx, mu_sel(:,n)); % , 'Color',col{n});
%     leg{end+1} = sprintf('Row V = %s',num2str(V(n)));
end
plot(xx, yN,'r','linewidth',3);
plot(xx(loc1),yMx,'bo','linewidth',3)

xlabel('\theta')
ylabel('Normalized f(\theta)')
title('Preference function')

[ymax, loc1] = max(mu2D);
[ymin, loc2] = min(mu2D);
plot(xx(loc1), ymax, 'bo')
plot(xx(loc2), ymin, 'rx')
legend(leg,'Location','best')
% figure
% plot(loc2)