
% load data 
data = importdata('output_pf_diagnistics_loglik.csv');
loglik = data.data; 
data = importdata('output_pf_diagnistics_weigths.csv');
weigths = data.data; 
data = importdata('output_pf_diagnistics_particles.csv');
particles = data.data; 
data = importdata('output_pf_diagnistics_Z_X.csv');
processes = data.data; % processes = [Z X]

% loglik 
mean_loglik = mean(loglik)
std_loglik = std(loglik)
figure

subplot(221) 
h1 = histogram(loglik,'FaceColor', 'r'); 
title('Histogram of loglik')

% particels 

std_particles = std(particles);

mean_std_particles = mean(std_particles)
std_std_std_particles = std(std_particles)

% plot hist of std for particles 
subplot(222) 
h1 = histogram(std_particles,'FaceColor', 'r'); 
title('Std of particles')


% plot particels and X process 
subplot(223) 
plot(particles')
hold on 
plot(processes(:,1), 'k','LineWidth',0.7)

title('Trajectories of particles and X process')

% plot weigths 

subplot(224) 
hold on 

for i = 1:1:size(weigths,1)
    h1 = histogram(weigths(i,:)); 
   
end
title('Histogram for weights')


% plot data  

figure
plot(processes(:,1))
title('Data')


