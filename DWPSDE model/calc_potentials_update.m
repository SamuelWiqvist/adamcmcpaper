% We will here only consider the power version of the potential function 


%% Using the old data set: 

% Load data 

% load data 
load_data 

% set parameters 
kappa = 0.3;
gamma = 0.9;
B = 1.;
c = 28.5;
d =  4; % was 4.
A = 0.01; % was 0.01
f = 0.;
g = 0.03; % was 0.03
power1 = 1.5;
power2 = 1.8;
sigma =  1.9;


B = 1.
c = 28.5
d =  4.
A = 0.01
f = 0.
g = 0.03
power1 = 1.5
power2 = 1.8
sigma =  1.9


% Plot data 

% plot data 
figure
plot(index, Z)

figure
hist(Z, 100)



%% Using the new data set 

% load data 
load_new_data_set

% set parameters 
kappa = 0.7;
gamma = 2.1;
B = 1.;
c = 22.5;
d =  13;
A = -0.0025;
f = 0.;
g = 0.;
power1 = 1.3;
power2 = 1.3;
sigma =  2.6;

% Plot data 

% linear transformation of data to obtain a scaling where it is easier to 
% construct the dwp model 

Z = 50*Z; 

Z = Z(1:100:length(Z)); 

Z = Z(1:24800);

index = 1:length(Z);

% plot data 
figure
plot(index, Z)

figure
hist(Z, 100)

%%
% define functions 

% potential 
V = @(x,B,c,d,A,f,g,power1,power2) abs(( abs(c - B*x).^(power1)/2 - d + g.*x)).^(power2)/(2) - f*x + (A*x.^2)./2;


% derivitive of potentential 
V_prime = @(x,B,c,d,A,f,g,power1,power2) A.*x - f + (power2*abs(abs(c - B.*x).^power1./2 - d + g.*x).^(power2 - 1).*sign(abs(c - B.*x).^power1/2 - d + g.*x).*(g - (B*power1*abs(c - B.*x).^(power1 - 1).*sign(c - B.*x))./2))./2;



% long term non eq dist
rho = @(x,B,c,d,A,f,g,power1,power2,sigma)  exp(-2/sigma^2 .* V(x,B,c,d,A,f,g,power1,power2)); 




% Compute long term probabilities 
% nomalizing constant 
C_roh = 1/integral(@(x)rho(x,B,c,d,A,f,g,power1, power2,sigma),-100,100); 

% normalized long term eq dist
roh_normalized = @(x,B,c,d,A,f,g,power1,power2,sigma) C_roh * rho(x,B,c,d,A,f,g,power1,power2,sigma);  

% integral of normalized long term eq dist
integral_long_eq_dist_normalized = integral(@(x)roh_normalized(x,B,c,d,A,f,g,power1, power2,sigma),-100,100); % the dist f_n should integrate to 1 


% Plotting set up 

% grid 
x = linspace(0,100,1000);
x_grid_potential = linspace(0,100,1000);

%  plot potential
figure
plot(x_grid_potential,V(x_grid_potential,B,c,d,A,f,g,power1,power2))
title('Potential')

%  plot derivitive of potential

figure
plot(x_grid_potential,V_prime(x_grid_potential,B,c,d,A,f,g,power1,power2))
title('Deriv. of Potential')

% Plot long term probability dist 

figure
plot(x_grid_potential, rho(x_grid_potential,B,c,d,A,f,g,power1, power2,sigma),'b','LineWidth',1.2)
title('Long term prob dist')

figure
plot(x_grid_potential, roh_normalized(x_grid_potential,B,c,d,A,f,g,power1, power2,sigma),'b','LineWidth',1.2)
title('Long term prob dist (normalized)')




% Plot long term probability dist and data 


% calc hist 
[f_Z,EDGES] = histcounts(Z,100);
x_Z = EDGES(1:100) + diff(EDGES);
%calc_area_hist(f_Z/trapz(x_Z,f_Z), EDGES) % checka area for normalized hist 

% plot nomalized long term equlibrium dist and nomalized hist of
% transformed data

figure
%bar(x_Z,f_Z/trapz(x_Z,f_Z)) % normalized histogram is this really correct?? 
bar(x_Z,f_Z/trapz(x_Z,f_Z))
hold on 
plot(x_Z, roh_normalized(x_Z,B,c,d,A,f,g,power1, power2,sigma),'b','LineWidth',1.2)
title('Long term prob. dist fit')

[f_kd,xi] = ksdensity(Z);

figure
%bar(x_Z,f_Z/trapz(x_Z,f_Z)) % normalized histogram is this really correct?? 
%plot(xi,f, 'r')
hold on 
plot(x_Z, roh_normalized(x_Z,B,c,d,A,f,g,power1, power2,sigma),'b','LineWidth',1.2)
title('Long term prob. dist fit')
hold on 
plot(xi,f_kd, 'r')


%%


V_emp = log(f_kd)*-sigma^2/2; 

figure
plot(V_emp)

figure
plot(diff(V_emp))


%%

p1 =   2.707e-06;
p2 =  -0.0005959;
p3 =     0.05017;
p4 =      -1.741;
p5 =       23.81;
       
       
V_4_order_poly = @(x, p1, p2, p3, p4, p5) p1*x.^4 + p2*x.^3 + p3*x.^2 + p4*x + p5

V_prime_4_order_poly =  @(x, p1, p2, p3, p4, p5) 4*p1*x.^3 + 3*p2*x.^2 + 2*p3*x + p4




x_grid_potential = linspace(0,100,1000);

%  plot potential
figure
plot(x_grid_potential,V_4_order_poly(x_grid_potential, p1, p2, p3, p4, p5))
title('Potential')

%  plot derivitive of potential

figure
plot(x_grid_potential,V_prime_4_order_poly(x_grid_potential, p1, p2, p3, p4, p5))
title('Deriv. of Potential')
