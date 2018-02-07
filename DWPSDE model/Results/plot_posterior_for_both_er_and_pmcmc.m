%addpath('results')

load('mcwm_est5_R10000_Sep_20_17.mat')% some saved matlab workspace

Theta_pmcmc = Theta;
burn_in_pmcmc = 5000;

load('da_gp_mcmc_est5_R10000_Sep_20_17')% some saved matlab workspace

Theta_ergp = Theta;
burn_in_ergp= 1; % both burn-in and training part 

% interesting runs:
% est2_PMCMC
% pmcmc_est2_R5000_May_29_17
% ergp_R5000_25_direct_May_29

%pmcmc_est5_R5000_May_29_17
%ergp_est5_R5000_25_direct_May_29
% Time: 
% mcmc: 11162/11/60 \approx 16.9 min/1000 iter 
% ergp: 3154/5/60 \approx 10.51 min/1000 iter 
% ergp stats:
% Time pre-er:  11162
% Time fit GP model:  442
% Time er-part:  3154
% Number early-rejections: 2375
% Secound stage direct limit: 100.000000
% Number of left-tail obs. with direct run of stage 2: 1289

% pmcmc_est5_R5000_May_29_17
% ergp_est5_R5000_25_direct_May_29


fontsize = 16;
linewidth = 1.5;  
%%

if N == 6
    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log c    '; 'log d    '];
    title_vec = [ 'Kappa'; 'Gamma'; 'c    '; 'd    '];
elseif N == 4
    title_vec_log = [ '$\log c$'; '$\log d$' ];
    title_vec = [ '$c$'; '$d$' ];
elseif N == 5
    title_vec_log = [ 'log A';'log c'; 'log d' ];
    title_vec = [ 'A';'c'; 'd' ];
elseif N == 8
    title_vec_log = [ 'log A    '; 'log c    '; 'log d    '; 'log p_1  '; 'log p_1  '; 'log sigma'];
    title_vec = [  'A    '; 'c    '; 'd    '; 'p_1  '; 'p_1  '; 'sigma'];
elseif N == 7
    title_vec_log = [ '$\log \kappa$'; '$\log \gamma$'; '$\log c     $'; '$\log d     $'; '$\log \sigma$'];
    title_vec = [ '$\Kappa$'; '$\gamma$'; '$c     $'; '$d     $'; '$\sigma$'];
elseif N == 9
    title_vec_log = [ 'log \kappa'; 'log \gamma'; 'log c     '; 'log d     '; 'log p_1   '; 'log p_1   '; 'log \sigma'];
    title_vec = [  '\kappa'; '\gamma'; 'c     '; 'd     '; 'p_1   '; 'p_1   '; '\sigma'];
else
    title_vec_log = [ 'log \kappa'; 'log \gamma'; 'log A     '; 'log c     '; 'log d     '; 'log g     '; 'log p_1   '; 'log p_1   '; 'log \sigma'];
    title_vec = [ '\kappa'; '\gamma'; 'A     '; 'c     '; 'd     '; 'g     '; 'p_1   '; 'p_1   '; '\sigma'];
end



%% 

% fix all xlabels and ylabels! useing
% xlabel('$\alpha$','Interpreter','LaTex') 
figure
for i = 1:N-2
    subplot(N-2,1,i)
    %[f, x] = hist(Theta(burn_in:end,i),50);
    %plot(x,f./trapz(x,f));
    [f_pmcmc,xi_pmcmc] = ksdensity(Theta_pmcmc(burn_in_pmcmc:end,i));
    plot(xi_pmcmc,f_pmcmc, 'b','LineWidth',linewidth)    
    %[f_pmcmc,xi_pmcmc] = histcounts(Theta_pmcmc(burn_in_pmcmc:end,i),'Normalization','pdf');  
    %plot(xi_pmcmc(2:end),f_pmcmc, 'b')    
    hold on 
    [f_ergp,xi_ergp] = ksdensity(Theta_ergp(burn_in_ergp:end,i));
    plot(xi_ergp,f_ergp, 'r','LineWidth',linewidth)
    %[f_ergp,xi_ergp] = histcounts(Theta_ergp(burn_in_ergp:end,i),'Normalization','pdf');  
    %plot(xi_ergp(2:end),f_ergp, 'r')    
    line([theta_true(i) theta_true(i)], get(gca, 'ylim'), 'color', 'k','LineWidth',linewidth);
    
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
        plot(x_grid, normpdf(x_grid, data_prior_dist(i,1),data_prior_dist(i,2)),'g','LineWidth',linewidth);
        axis([x_grid(1) x_grid(end) -inf inf])
    end
    
    %title(strcat({'Marginal Posterior: '},{title_vec(i,:)}))
    %legend('posterior','true parameter value','prior','Location','eastoutside')
    %ylabel('Probability')
    ylabel(title_vec_log(i,:),'Interpreter','LaTex')
    set(gca,'fontsize',fontsize)
end
%xlabel('Iteration')
%suptitle('Posterior')
set(gca,'fontsize',fontsize)
%% 

nbr_acf_lags =  50;

acf_mcmc = zeros(N-2,nbr_acf_lags+1);
acf_ergp = zeros(N-2,nbr_acf_lags+1);

for i = 1:N-2
    [acf_mcmc(i,:),lags_acf_mcmc,bounds_acf_mcmc] = autocorr(Theta_pmcmc(burn_in:end,i),nbr_acf_lags);
    [acf_ergp(i,:),lags_acf_ergp,bounds_acf_ergp] = autocorr(Theta_ergp(burn_in_ergp:end,i),nbr_acf_lags);
end


figure
for i = 1:N-2
    subplot(N-2,1,i)
    plot(lags_acf_mcmc, acf_mcmc(i,:), '--b')
    hold on 
    plot(lags_acf_ergp, acf_ergp(i,:), '--r')
    title(strcat({'Acf for '}, {title_vec_log(i,:)}),'Interpreter','LaTex')
end
xlabel('Lag')

%% Compute ESS_min / sec (old method)
Theta_ess = Theta_pmcmc; % Theta_pmcmc % Theta_ergp 
burn_in_ess = burn_in_pmcmc; % pmcmc: burn_in_pmcmc, dagpmcmc: burn_in_ergp
time_sec = 11162/11*5; % pmcmc: 11162/11*5, dagpmcmc: 3154
n = length(Theta_ess) - burn_in_ess; 

nbr_acf_lags =  1001;

acf = zeros(N-2,nbr_acf_lags+1);

for i = 1:N-2
    [acf(i,:),lags_acf,bounds_acf] = autocorr(Theta_ess(burn_in_ess:end,i),nbr_acf_lags);
end

ESS = n./(1+2*sum(acf(:,2:end),2))

ESS_min = min(ESS)

ESS_min_over_time = ESS_min/time_sec

%% Compute ESS_min/sec (batch method)
% according to paper: https://arxiv.org/pdf/1403.5536.pdf 
Theta_ess = Theta_pmcmc; % Theta_pmcmc % Theta_ergp 
burn_in_ess = burn_in_pmcmc; % pmcmc: burn_in_pmcmc, dagpmcmc: burn_in_ergp
time_sec = 11162/11*5; % pmcmc: 11162/11*5, dagpmcmc: 3154
n = length(Theta_ess) - burn_in_ess; 
% g is set to NULL 
Z = mean(Theta_ess,1);
lambda2 = var(Theta_ess,1);
tau = 1/2;
b_n = floor(n^tau); 
a_n = floor(n/b_n); 
sigma2 = zeros(size(lambda2));
for j = 1:a_n 
    Y_j = zeros(size(lambda2));
    for i = (j-1)*b_n:j*b_n-1
        Y_j = Y_j + Theta_ess(i+1,:);
    end 
    Y_j = 1/b_n*Y_j;
    sigma2 = sigma2 + (Y_j - Z).^2;
end 
sigma2 = (b_n)/(a_n-1)*sigma2;

ESS = n*lambda2./sigma2

ESS_min = min(ESS)

ESS_min_per_sec = ESS_min/time_sec














