
clc
clear all
close all

load_data_from_files = true; % load data from files or form some matlab workspace
plot_prior_trace_plot = false; 
ergp = '_ergp'; % set to _ergp to load ER-GP file  o.w. []
jobname = 'est7_betamh_01ada_gp_mcmc'; % set to jobname string  


% gp_training_7_par
% est7_betamh_01da_gp_mcmc

% runs; 
% old_data_set_new_dt_est_2_param_N_25
% old_data_set_new_dt_est_2_param_N_200
% new_data_set_est_2_N_25
% new_data_set_est_2_N_200

% old_data_set_new_dt_est_5_param_N_25
% old_data_set_new_dt_est_5_param_N_200

% new_data_set_est_5_N_25

% old_data_set_est_7_N_100
% old_data_set_est_7_N_200_run_2
% new_data_set_est_7_N_100

% new_data_set_est_9_N_100
% old_data_set_est_9_N_100

% old_real_data_set_est_5_N_200
% old_real_data_set_est_7_N_200

% R5000_N25
% R5000_N25_25_direct_MH
% est9_R12000_MCWM
% est9_R12000_MCWM_N100
% est2_R5000_PMCMC
% est9_starting_at_theta_true
% est5_R5000_N25_desktop
% _ergp est5_R5000_N25_25_direct_MH_desktop


% da_2_param
% ada_2_param
% da_5_param
% ada_5_param

% da_gp_mcmc_est_2
% ada_gp_mcmc_est_2

% da_gp_mcmc_est_5
% ada_gp_mcmc_est_5

% da_gp_mcmc_est_7
% ada_gp_mcmc_est_7

% 7/10 da/ada runs on lunrac: 

% old_data_set_new_dt_est_2_da_ada_run: outputs_208257
% da_est2_gp_mcmc: outputs_208255
% ada_est2_gp_mcmc: outputs_208256

% old_data_set_new_dt_est_5_da_ada_run: outputs_208995
% da_est5_gp_mcmc: outputs_208996
% ada_est5_gp_mcmc: outputs_208256

% old_data_set_new_dt_est_7_da_ada_run: outputs_209000
% da_est7_gp_mcmc: outputs_208999
% ada_est7_gp_mcmc: outputs_208998

% adagpMCMC_training_dataada_est7_gp_mcmc
% dagpMCMC_training_datada_est7_gp_mcmc
% dagpMCMC_training_datada_est5_gp_mcmc


% simulations using same training data

% da_ada_gpMCMC_training_dataest2_param
% est2_paramda_gp_mcmc
% est2_paramada_gp_mcmc

% da_ada_gpMCMC_training_dataest5_param
% est5_paramda_gp_mcmc
% est5_paramada_gp_mcmc

% da_ada_gpMCMC_training_dataest7_param
% est7_paramda_gp_mcmc
% est7_paramada_gp_mcmc


% setting beta_mh = 0 

% est2_test_beta_mh_zeroda_gp_mcmc

% results for paper 1:

% mcwm_res 

% test logsumexp fix 
% mcwm_est_7_expsum_fix
if load_data_from_files

    data_res = importdata(strcat('output_res',ergp,jobname,'.csv'));

    data_res = data_res.data;
    [ M , N ] = size(data_res);

    data_param = importdata(strcat('output_param',ergp,jobname,'.csv'));
    data_param = data_param.data;
    theta_true = data_param(1:N-2);
    burn_in = data_param(N-2+1);

    data_prior_dist = importdata(strcat('output_prior_dist',ergp,jobname,'.csv'));
    data_prior_dist = data_prior_dist.data;

    data_prior_dist_type = importdata(strcat('output_prior_dist_type',ergp,jobname,'.csv'));
    data_prior_dist_type = data_prior_dist_type{2,1};
    data_prior_dist_type = data_prior_dist_type(6:end-1);

    Z = importdata(strcat('data_used',ergp,jobname,'.csv'));
    Z = Z.data;
    Z = Z(:,1); 
        
else
    
    addpath('results')
    %load('pmcmc_est5_R5000_May_29_17.mat')
    load('ergp_est5_R5000_25_direct_May_29.mat')% some saved matlab workspace
    
    % interesting runs:
    %res_est_4_normal_prior_AM_gen_April_29_17    
    %res_est_5_normal_prior_AM_gen_April_29_17     
    %res_6_param_normal_prior_AM_gen_May_3_17
    %res_7_param_normal_prior_May_3_17
    %res_est_9_param_May_3_17
    
end


Theta = data_res(:,1:N-2);
loglik = data_res(:,N-1);
accept_vec = data_res(:,N);

accept_rate = sum(accept_vec)/M;
nbr_acf_lags =  50;

acf = zeros(N-2,nbr_acf_lags+1);

if N == 6
    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log c    '; 'log d    '];
    title_vec = [ 'Kappa'; 'Gamma'; 'c    '; 'd    '];
elseif N == 4
    title_vec_log = [ 'log c'; 'log d' ];
    title_vec = [ 'c'; 'd' ];
elseif N == 5
    title_vec_log = [ 'log A';'log c'; 'log d' ];
    title_vec = [ 'A';'c'; 'd' ];
elseif N == 8
    title_vec_log = [ 'log A    '; 'log c    '; 'log d    '; 'log p_1  '; 'log p_2  '; 'log sigma'];
    title_vec = [  'A    '; 'c    '; 'd    '; 'p_1  '; 'p_1  '; 'sigma'];
elseif N == 7
    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log c    '; 'log d    '; 'log sigma'];
    title_vec = [ 'Kappa'; 'Gamma'; 'c    '; 'd    '; 'sigma'];
elseif N == 9
    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log c    '; 'log d    '; 'log p_1  '; 'log p_2  '; 'log sigma'];
    title_vec = [  'Kappa'; 'Gamma'; 'c    '; 'd    '; 'p_1  '; 'p_1  '; 'sigma'];
else
    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log A    '; 'log c    '; 'log d    '; 'log g    '; 'log p_1  '; 'log p_2  '; 'log sigma'];
    title_vec = [ 'Kappa'; 'Gamma'; 'A    '; 'c    '; 'd    '; 'g    '; 'p_1  '; 'p_1  '; 'sigma'];
end

for i = 1:N-2
    [acf(i,:),lags_acf,bounds_acf] = autocorr(Theta(burn_in:end,i),nbr_acf_lags);
end


% plot trace plots 
figure
for i = 1:N-2
        subplot(N-2,1,i)
        plot(Theta(:,i))
        hold on
        plot(theta_true(i)*ones(1,length(Theta)), 'r--')
        if plot_prior_trace_plot
            plot(data_prior_dist(i,1)*ones(1,length(Theta)), 'g--')
            plot(data_prior_dist(i,2)*ones(1,length(Theta)), 'g--')
        end
        ylabel(title_vec_log(i,:))
        % no chartjunk! 
%         if plot_prior_trace_plot
%             legend('chain','true parameter value','prior','Location','eastoutside')
%         else
%             legend('chain','true parameter value','Location','eastoutside')
%         end
end
xlabel('Iteration')


figure
for i = 1:N-2
        subplot(N-2,1,i)
        plot(exp(Theta(:,i)))
        hold on
        plot(exp(theta_true(i))*ones(1,length(Theta)), 'r--')
        if plot_prior_trace_plot
            plot(data_prior_dist(i,1)*ones(1,length(Theta)), 'g--')
            plot(data_prior_dist(i,2)*ones(1,length(Theta)), 'g--')
        end
        ylabel(title_vec(i,:))
        % no chartjunk! 
%         if plot_prior_trace_plot
%             legend('chain','true parameter value','prior','Location','eastoutside')
%         else
%             legend('chain','true parameter value','Location','eastoutside')
%         end
end
xlabel('Iteration')

% plot trace plots after burn in 
figure
x_axis = burn_in+1:size(Theta,1); 
for i = 1:N-2
        subplot(N-2,1,i)
        plot(x_axis, Theta(burn_in+1:end,i))
        hold on
        plot(x_axis, theta_true(i)*ones(1,length(Theta(burn_in+1:end,:))), 'r--')
        if plot_prior_trace_plot
            plot(x_axis, data_prior_dist(i,1)*ones(1,length(Theta(burn_in+1:end,:))), 'g--')
            plot(x_axis, data_prior_dist(i,2)*ones(1,length(Theta(burn_in+1:end,:))), 'g--')
        end
        ylabel(title_vec_log(i,:))
        % no chartjunk! 
%         if plot_prior_trace_plot
%             legend('chain','true parameter value','prior','Location','eastoutside')
%         else
%             legend('chain','true parameter value','Location','eastoutside')
%         end
end
xlabel('Iteration')


figure
for i = 1:N-2
        subplot(N-2,1,i)
        plot(x_axis, exp(Theta(burn_in+1:end,i)))
        hold on
        plot(x_axis, exp(theta_true(i))*ones(1,length(Theta(burn_in+1:end,:))), 'r--')
        if plot_prior_trace_plot
            plot(x_axis, data_prior_dist(i,1)*ones(1,length(Theta(burn_in+1:end,:))), 'g--')
            plot(x_axis, data_prior_dist(i,2)*ones(1,length(Theta(burn_in+1:end,:))), 'g--')
        end
        ylabel(title_vec(i,:))
        % no chartjunk! 
%         if plot_prior_trace_plot
%             legend('chain','true parameter value','prior','Location','eastoutside')
%         else
%             legend('chain','true parameter value','Location','eastoutside')
%         end
end
xlabel('Iteration')


figure
for i = 1:N-2
    subplot(N-2,1,i)
    plot(lags_acf, acf(i,:))
    title(strcat({'Acf for '}, {title_vec_log(i,:)}))
    xlabel('Lag')
end


figure
for i = 1:N-2
    subplot(N-2,1,i)
    %[f, x] = hist(Theta(burn_in:end,i),50);
    %plot(x,f./trapz(x,f));
    [f,xi] = ksdensity(Theta(burn_in:end,i));
    plot(xi,f)

    hold on
    line([theta_true(i) theta_true(i)], get(gca, 'ylim'), 'color', 'r');
    
    if strcmp(data_prior_dist_type,'Uniform') 
        if data_prior_dist(i,1) < 0
           start_val = data_prior_dist(i,1)*1.05;
        elseif data_prior_dist(i,1) > 0
            start_val = data_prior_dist(i,1)*0.95;
        else 
            start_val = -0.05;
        end
        if data_prior_dist(i,2) < 0
            end_val = data_prior_dist(i,2)*.95;
        elseif data_prior_dist(i,2) > 0
            end_val = data_prior_dist(i,2)*1.05;
        else
            end_val = 0.05;
        end

        x_grid = start_val:0.001:end_val;
        plot(x_grid, unifpdf(x_grid,data_prior_dist(i,1), data_prior_dist(i,2)), 'g')
        axis([x_grid(1) x_grid(end) -inf inf])
    elseif strcmp(data_prior_dist_type,'Normal')
        x_grid = (data_prior_dist(i,1)-4*data_prior_dist(i,2)):0.001:(data_prior_dist(i,1)+4*data_prior_dist(i,2));
        plot(x_grid, normpdf(x_grid, data_prior_dist(i,1),data_prior_dist(i,2)),'g');
        axis([x_grid(1) x_grid(end) -inf inf])
    end
    
    %title(strcat({'Marginal Posterior: '},{title_vec(i,:)}))
    %legend('posterior','true parameter value','prior','Location','eastoutside')
    %ylabel('Probability')
    title(title_vec_log(i,:))
end
    
figure
for i = 1:N-2
    subplot(N-2,1,i)
    %[f, x] = hist(exp(Theta(burn_in:end,i)),50);
    [f,xi] = ksdensity(exp(Theta(burn_in:end,i)));
    plot(xi,f)
    hold on
    line([exp(theta_true(i)) exp(theta_true(i))], get(gca, 'ylim'), 'color', 'r');
    title(title_vec(i,:))
    %ylabel('Density')
end    

figure
plot(loglik)
ylabel('Log-likelhood')
xlabel('Iteration')

% plot acceptance rate for each K:th iteration
k = 500; 

accept_vec_k = zeros(length(Theta)/k ,1);
intervals = zeros(length(accept_vec_k),2);

j = 1;
for r = 2:length(Theta)
    if mod(r-1,k) == 0
        accept_vec_k(j) = sum(accept_vec(r-k:r-1))/( r-1 - (r-k) );
        intervals(j,:) = [r-k, r-1];
        j = j +1;
    end
end
accept_vec_k(end) = sum(accept_vec(length(accept_vec)-k+1:end))/( k );

intervals(length(Theta)/k,:) = [length(accept_vec)-k+1, length(accept_vec)];

intervals_vec = {'start'};

for j = 1:length(intervals)
    intervals_vec(j) = {strcat(num2str(intervals(j,1)), '-', num2str(intervals(j,2)))}; % I should probably use vertical concatunate here!
    %intervals_vec(j) = {vertcat(intervals(j,1), intervals(j,2))}; 
end

figure
bar(k:k:length(Theta),accept_vec_k)
%set(gca,'XTickLabel',intervals_vec)
axis([0 length(Theta)+k 0 inf])
ylabel('Acceptance rate')
xlabel('Iteration')



% plot simulated data
figure
plot(1:length(Z),Z)
axis([0 length(Z), -inf inf])
xlabel('Index')
%title('Data')
set(gca,'FontSize',14)

figure
hist(Z,50)
%title('Histogram')
set(gca,'FontSize',14)


% calc estiamted parameters
disp('True parameter value:')
exp(theta_true)
disp('Parameter estimations (posterior mean log-transformed):')
exp(mean(Theta(burn_in+1:end,:)))
disp('Parameter estimations (posterior std on log-scale):')
std(Theta(burn_in+1:end,:))

% parameter estiamtions 
disp('Parameter estimation:')

disp('True parameter value (log-scale):')
round(theta_true,3)

disp('Posterior mean:')
round(mean(Theta(burn_in+1:end,:)),3)

disp('Posterior quantile interval (2.5th and 97.5th quantiles):')
round(quantile(Theta(burn_in+1:end,:), [.025,  .975]),3)

%% Save results 
filename = 'ada_res'; 

save(filename)
