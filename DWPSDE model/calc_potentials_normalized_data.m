% We will here only consider the power version of the potential function 

% We will employ normalization to transform all data sets into the same
% data range 

% plots for unnoarmalized and normalized
% data: unnormalized_and_normalized_data.png 

%% Start scritp 

close all
clear all
clc

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


B = 1.;
c = 28.5;
d =  4.;
A = 0.01;
f = 0.;
g = 0.03;
power1 = 1.5;
power2 = 1.8;
sigma =  1.9; 


% Plot data 

% plot data 
figure
subplot(121)
plot(index, Z)
axis([0 length(Z), -inf inf])
title('Data')
subplot(122)
hist(Z, 100)
title('Histogram of data')



%% Using the new data set 


clear all
clc


% load data 
load_new_data_set
index = 1:length(Z); 

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

% plot data 
figure
subplot(121)
plot(index, Z)
axis([0 length(Z), -inf inf])
title('Data')
subplot(122)
hist(Z, 100)
title('Histogram of data')


%% Transformed data 

Z = (Z-mean(Z))./std(Z); 
%Z = Z + max(-min(Z), max(Z));

figure
subplot(121)
plot(index, Z)
axis([0 length(Z), -inf inf])
title('Data (Transformed)')
subplot(122)
hist(Z, 100)
title('Histogram of data  (Transformed)')

%% Parameters for old standardized data 

kappa = 0.3;
gamma = 0.9;
d = 1.5; 
A = -0.03; 
g = -0.07; 
power1 = 1.6; % 1.6
power2 = 1.9; % 1.8
sigma =  1.6;
%% Parameters for new standardized data 

kappa = 0.3;
gamma = 0.9;
d = 2; % was 4.
A = 2; % was 0.01
g = 0; % was 0.03
power1 = 1.6;
power2 = 1.8;
sigma =  0.5;
m = max(Z); 

kappa = 0.3;
gamma = 0.9;
d = 1.3; 
A = 0.015; 
g = 0.01; 
power1 = 1.6; % 1.6
power2 = 1.8; % 1.8
sigma =  .6;
m = max(Z); 

%% Compute potentials and find ok fit to long term prob dist 


% define functions 

% potential 
V = @(x,c,d,A,f,g,power1,power2) abs(( abs(c - B*x).^(power1)/2 - d + g.*x)).^(power2)/(2) - f*x + (A*x.^2)./2;


% derivitive of potentential 
% V_prime = @(x,B,c,d,A,f,g,power1,power2) A.*x - f + (power2*abs(abs(c - B.*x).^power1./2 - d + g.*x).^(power2 - 1).*sign(abs(c - B.*x).^power1/2 - d + g.*x).*(g - (B*power1*abs(c - B.*x).^(power1 - 1).*sign(c - B.*x))./2))./2;


V_prime = @(x,c,d,A,f,g,power1,power2)  A.*x - f + (power2*abs(abs(c - B.*x).^power1./2 - d + g.*x).^(power2 - 1).*sign(abs(c - B.*x).^power1/2 - d + g.*x).*(g - (B*power1*abs(c - B.*x).^(power1 - 1).*sign(c - B.*x))./2))./2;


% long term non eq dist
rho = @(x,c,d,A,f,g,power1,power2,sigma)  exp(-2/sigma^2 .* V(x,c,d,A,f,g,power1,power2)); 




% Compute long term probabilities 
% nomalizing constant 
C_roh = 1/integral(@(x)rho(x,c,d,A,f,g,power1,power2,sigma),-100,100); 

% normalized long term eq dist
roh_normalized = @(x,c,d,A,f,g,power1,power2,sigma) C_roh * rho(x,c,d,A,f,g,power1,power2,sigma);  

% integral of normalized long term eq dist
integral_long_eq_dist_normalized = integral(@(x)roh_normalized(x,c,d,A,f,g,power1,power2,sigma),-100,100); % the dist f_n should integrate to 1 


% Plotting set up 

% grid 
%x = linspace(0,100,1000);
x = linspace(-4,4,1000); % since we are on the normalized scale now 
%x_grid_potential = linspace(0,100,1000);
x_grid_potential = linspace(min(Z)-1,max(Z)+1,1000); % since we are on the normalized scale now 
 
%  plot potential
figure
plot(x_grid_potential,V(x_grid_potential,c,d,A,f,g,power1,power2))
title('Potential')

%  plot derivitive of potential

figure
plot(x_grid_potential,V_prime(x_grid_potential,c,d,A,f,g,power1,power2))
title('Deriv. of Potential')

% Plot long term probability dist 

figure
plot(x_grid_potential, rho(x_grid_potential,c,d,A,f,g,power1,power2,sigma),'b','LineWidth',1.2)
title('Long term prob dist')

figure
plot(x_grid_potential, roh_normalized(x_grid_potential,c,d,A,f,g,power1,power2,sigma),'b','LineWidth',1.2)
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
plot(x_Z, roh_normalized(x_Z,c,d,A,f,g,power1,power2,sigma),'b','LineWidth',1.2)
title('Long term prob. dist fit')



