


% true values      prior dist             comment 
kappa = 0.3;    % inv-gamma(3,1.5)      uninformative  
gamma = 0.9;    % inv-gamma(3,1.5)      uninformative
A = 0.01;       % inv-gamma(3,0.05)     uninformative
c = 28.5;       % normal(28,2)          informative
d =  4.;        % normal(4,1)           informative
g = 0.03;       % inv-gamma(3,0.05)     uninformative
power1 = 1.5;   % gamma(2,2)            uninformative
power2 = 1.8;   % gamma(2,2)            uninformative
sigma =  1.9;   % gamma(2,2)            uninformative
  
% plottong true values and priors

figure

% plot kappa 
subplot(911)
x = 0:0.01:2;
hold on 
plot(x,invgampdf(x,3,1.5))
line([kappa kappa], get(gca, 'ylim'), 'color', 'r');
ylabel('kappa')

% plot gamma 
subplot(912)
x = 0:0.01:2;
hold on 
plot(x,invgampdf(x,3,1.5))
line([gamma gamma], get(gca, 'ylim'), 'color', 'r');
ylabel('gamma')

% plot c 
subplot(913)
x = 22:0.01:33;
hold on 
plot(x,normpdf(x,28,2))
line([c c], get(gca, 'ylim'), 'color', 'r');
ylabel('c')

% plot d
subplot(914)
x = 1:0.01:7;
hold on 
plot(x,normpdf(x,4,1))
line([d d], get(gca, 'ylim'), 'color', 'r');
ylabel('d')

% plot A
subplot(915)
x = 0:0.001:0.5;
hold on 
plot(x,invgampdf(x,3,0.05))
line([A A], get(gca, 'ylim'), 'color', 'r');
ylabel('A')

% plot g
subplot(916)
x = 0:0.001:0.5;
hold on 
plot(x,invgampdf(x,3,0.05))
line([g g], get(gca, 'ylim'), 'color', 'r');
ylabel('g')

% plot p1
subplot(917)
x = 0:0.01:10;
hold on 
plot(x,gampdf(x,2,2))
line([power1 power1], get(gca, 'ylim'), 'color', 'r');
ylabel('p1')

% plot p2
subplot(918)
x = 0:0.01:10;
hold on 
plot(x,gampdf(x,2,2))
line([power2 power2], get(gca, 'ylim'), 'color', 'r');
ylabel('p2')

% plot sigma
subplot(919)
x = 0:0.01:10;
hold on 
plot(x,gampdf(x,2,2))
%plot(x,invgampdf(x,4,7))
line([sigma sigma], get(gca, 'ylim'), 'color', 'r');
ylabel('sigma')

