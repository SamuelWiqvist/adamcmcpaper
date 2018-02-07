
load_data_from_files = true; % load data from files or form some matlab workspace
plot_prior_trace_plot = false; 

if load_data_from_files
    data_res = importdata('output_res.csv');
    data_res = data_res.data;
    [ M , N ] = size(data_res);
    
    data_param = importdata('output_param.csv');
    data_param = data_param.data;
    theta_true = data_param(1:N-2);
    burn_in = data_param(N-2+1);
    
    data_prior_dist = importdata('output_prior_dist.csv');
    data_prior_dist = data_prior_dist.data;
    
    data_prior_dist_type = importdata('output_prior_dist_type.csv');
    data_prior_dist_type = data_prior_dist_type{2,1};
    data_prior_dist_type = data_prior_dist_type(6:end-1);
    
    Z = importdata('data_used.csv');
    Z = Z.data;
    Z = Z(:,1); 
else
    addpath('results')
    load('res_est_3_par_nonlog_prior_15_2_17') % some saved matlab workspace
end

Theta = data_res(:,1:N-2);
loglik = data_res(:,N-1);
accept_vec = data_res(:,N);

accept_rate = sum(accept_vec)/M;
nbr_acf_lags =  10;

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
    title_vec_log = [ 'log A    '; 'log c    '; 'log d    '; 'log p_1  '; 'log p_1  '; 'log sigma'];
    title_vec = [  'A    '; 'c    '; 'd    '; 'p_1  '; 'p_1  '; 'sigma'];
elseif N == 7
    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log c    '; 'log d    '; 'log sigma'];
    title_vec = [ 'Kappa'; 'Gamma'; 'c    '; 'd    '; 'sigma'];
elseif N == 9
    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log c    '; 'log d    '; 'log p_1  '; 'log p_1  '; 'log sigma'];
    title_vec = [  'Kappa'; 'Gamma'; 'c    '; 'd    '; 'p_1  '; 'p_1  '; 'sigma'];
else

    title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log A    '; 'log c    '; 'log d    '; 'log g    '; 'log p_1  '; 'log p_1  '; 'log sigma'];
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
%         ylabel(title_vec_log(i,:))
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
    [f,xi] = ksdensity(Theta(burn_in:end,i));
    plot(xi,f)

    hold on
    line([theta_true(i) theta_true(i)], get(gca, 'ylim'), 'color', 'r');
    
    
 
    
    ylabel(title_vec_log(i,:))
end

figure
for i = 1:N-2
    subplot(N-2,1,i)
    [f,xi] = ksdensity(exp(Theta(burn_in:end,i)));
    plot(xi,f)
    hold on
    if N == 6

        if i == 2
            x = 22:0.01:33;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 3
            x = 0:0.01:10;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)), 'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 1
            x = 0:0.001:0.1;
            plot(x,invgampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 4
            x = 0:0.001:5;
            plot(x,invgampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        end

    elseif N == 7

        if i == 4
            x =  1:0.01:7;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 5
            x = 0:0.001:2;
            plot(x,gampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)), 'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 3
            x = 22:0.01:33;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 1  || i == 2 
            x = 0:0.01:2;
            plot(x,gampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        end

     elseif N == 9

        if i == 4
            x =  1:0.01:7;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 3
            x = 22:0.01:33;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 1  || i == 2 || i == 5 || i == 6 || i == 7 
            x = 0:0.01:2;
            plot(x,gampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        end

        
    elseif N == 4
%         if i == 1
%              x = 0:0.01:10;
%             plot(x,gampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)), 'g')
%             line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
%         elseif i == 2
%             x = 0:0.01:10;
%             plot(x,gampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
%             line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');        
%         end


        if i == 1
             x = 22:0.01:33;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)), 'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 2
            x = 1:0.01:7;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');        
        end
    elseif N == 5
        if i == 2
            x = 22:0.01:33;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 3
            x = 0:0.01:10;
            plot(x,normpdf(x,data_prior_dist(i,1),data_prior_dist(i,2)), 'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');
        elseif i == 1
            x = 0:0.001:0.1;
            plot(x,invgampdf(x,data_prior_dist(i,1),data_prior_dist(i,2)),'g')
            line(exp([theta_true(i) theta_true(i)]), get(gca, 'ylim'), 'color', 'r');        
        end
    else
        %title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log A    '; 'log c    '; 'log d    '; 'log g    '; 'log p_1  '; 'log p_1  '; 'log sigma'];
        %title_vec = [ 'Kappa'; 'Gamma'; 'A    '; 'c    '; 'd    '; 'g    '; 'p_1  '; 'p_1  '; 'sigma'];
    end
    ylabel(title_vec(i,:))


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

%mean(exp(Theta(burn_in:end,:)))
%exp(theta_true)

% calc estiamted parameters
disp('True parameter value:')
exp(theta_true)
disp('Parameter estimations (posterior mean log-transformed):')
exp(mean(Theta(burn_in+1:end,:)))
disp('Parameter estimations (posterior std on log-scale):')
std(Theta(burn_in+1:end,:))

