
using Plots
using PyPlot
using StatPlots
using KernelDensity
using Distributions

# load functions to compute posterior inference
if Sys.CPU_CORES == 8
  include("C:\\Users\\samuel\\Dropbox\\Phd Education\\Projects\\project 1 accelerated DA and DWP SDE\\code\\utilities\\posteriorinference.jl")
else
  include("C:\\Users\\samue\\OneDrive\\Documents\\GitHub\\adamcmcpaper\\utilities\\posteriorinference.jl")
end

# text and lable size
text_size = 15
label_size = 15


load_data_from_files = true # % load data from files or form some matlab workspace
plot_prior_trace_plot = false
ergp = ""# % set to _ergp to load ER-GP file  o.w. []
jobname = "test_new_calc_for_a" # set to jobname string


if load_data_from_files

    data_res = readtable("output_res"*ergp*jobname*".csv")

    data_res = data_res.data
    [ M , N ] = size(data_res)

    data_param = readtable("output_param"*ergp*jobname*".csv")
    data_param = data_param.data
    theta_true = data_param[1:N-2]
    burn_in = data_param[N-2+1]

    data_prior_dist = readtable("output_prior_dist"*ergp*jobname*".csv")
    data_prior_dist = data_prior_dist.data;

    data_prior_dist_type = readtable("output_prior_dist_type"*ergp*jobname*".csv")
    data_prior_dist_type = data_prior_dist_type{2,1};
    data_prior_dist_type = data_prior_dist_type(6:end-1);

    Z = readtable("data_used"*ergp*jobname*".csv");
    Z = Z.data;
    Z = Z(:,1);

else

    # this option should be used to load from stored .jld files

end


Theta = data_res[:,1:N-2]
loglik = data_res[:,N-1]
accept_vec = data_res[:,N]

accept_rate = sum(accept_vec)/M
nbr_acf_lags =  50

acf = zeros(N-2,nbr_acf_lags+1)

# use L"$\log r$"

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
    [acf(i,:),lags_acf,bounds_acf] = autocor(Theta(burn_in:end,i),nbr_acf_lags);
end


acceptance_rate = sum(accept_vec[burn_in:end])/length(accept_vec[burn_in:end])


@printf "Accept rate: %.4f %% \n" acceptance_rate*100
@printf "Nbr outside of prior: %d  \n" nbr_out_side_prior

@printf "Posterior mean:\n"
Base.showarray(STDOUT,mean(Theta[:,burn_in+1:end],2),false)
@printf "\n"

@printf "Posterior standard deviation:\n"
Base.showarray(STDOUT,std(Theta[:,burn_in+1:end],2),false)
@printf "\n"

@printf "Posterior quantile intervals (2.5th and 97.5th quantiles as default):\n"
Base.showarray(STDOUT,calcquantileint(Theta[:,burn_in+1:end],lower_q_int_limit,upper_q_int_limit),false)
@printf "\n"

@printf "RMSE for parameter estimations:\n"
Base.showarray(STDOUT,RMSE(theta_true, Theta[:,burn_in+1:end]),false)
@printf "\n"



# plot trace plots
Pyplot.figure()
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
end
xlabel('Iteration')

# plot trace plots on non-log scale
Pyplot.figure()
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
end
xlabel('Iteration')

# plot trace plots after burn in
Pyplot.figure()
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

end
xlabel('Iteration')


Pyplot.figure()
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
end
xlabel('Iteration')


Pyplot.figure()
for i = 1:N-2
    subplot(N-2,1,i)
    plot(lags_acf, acf(i,:))
    title("Acf for " * title_vec_log(i,:))
    xlabel('Lag')
end


Pyplot.figure()
for i = 1:N-2
  subplot(N-2,1,i)
  [f,xi] = ksdensity(Theta(burn_in:end,i));
  plot(xi,f)

  hold on
  line([theta_true(i) theta_true(i)], get(gca, 'ylim'), 'color', 'r');

  if data_prior_dist_type == "Uniform"
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
  elseif data_prior_dist_type = "Normal"
      x_grid = (data_prior_dist(i,1)-4*data_prior_dist(i,2)):0.001:(data_prior_dist(i,1)+4*data_prior_dist(i,2));
      plot(x_grid, normpdf(x_grid, data_prior_dist(i,1),data_prior_dist(i,2)),'g');
      axis([x_grid(1) x_grid(end) -inf inf])
  end

  title(title_vec_log(i,:))
end

Pyplot.figure()
for i = 1:N-2
  subplot(N-2,1,i)
  [f,xi] = ksdensity(exp(Theta(burn_in:end,i)));
  plot(xi,f)
  hold on
  line([exp(theta_true(i)) exp(theta_true(i))], get(gca, 'ylim'), 'color', 'r');
  title(title_vec(i,:))
end

figure
plot(loglik)
ylabel('Log-likelhood')
xlabel('Iteration')

# plot acceptance rate for each K:th iteration
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
end

PyPlot.figure()
bar(k:k:length(Theta),accept_vec_k)
%set(gca,'XTickLabel',intervals_vec)
axis([0 length(Theta)+k 0 inf])
ylabel('Acceptance rate')
xlabel('Iteration')



# plot simulated data
PyPlot.figure()

plot(1:length(Z),Z)
axis([0 length(Z), -inf inf])
xlabel('Index')
%title('Data')
set(gca,'FontSize',14)

PyPlot.figure()
hist(Z,50)
%title('Histogram')
set(gca,'FontSize',14)

# save results to jld file 
filename = 'ada_res';

save(filename)
