# load packages

using Plots
using PyPlot
using StatPlots
using KernelDensity
using Distributions
using DataFrames

# set correct path
try
  cd("DWPSDE model")
catch
  warn("Already in the DWP-SDE folder.")
end

# set dir
try
    cd("Results")
catch
    warn("Already in the Results folder for the DWP model.")
end

# load functions to compute posterior inference
if Sys.CPU_CORES == 8
    include("C:\\Users\\samuel\\Dropbox\\Phd Education\\Projects\\project 1 accelerated DA and DWP SDE\\code\\utilities\\posteriorinference.jl")
else
    include("C:\\Users\\samue\\OneDrive\\Documents\\GitHub\\adamcmcpaper\\utilities\\posteriorinference.jl")
end

# text and lable size
text_size = 15
label_size = 15


load_data_from_files = true # load data from files or form some  workspace
dagp = "_dagp" #  set to _dagp to load ER-GP file  o.w. use ""
jobname = "lunarc_mcwm" # set to jobname string

if load_data_from_files

    data_res = convert(Array,readtable("output_res"*dagp*jobname*".csv"))

    M, N = size(data_res)

    data_param = convert(Array,readtable("output_param"*dagp*jobname*".csv"))

    theta_true = data_param[1:N-2]
    burn_in = Int64(data_param[N-2+1])

    data_prior_dist = convert(Array,readtable("output_prior_dist"*dagp*jobname*".csv"))

    data_prior_dist_type = convert(Array,readtable("output_prior_dist_type"*dagp*jobname*".csv"))
    data_prior_dist_type = data_prior_dist_type[2]

    Z = convert(Array,readtable("data_used"*dagp*jobname*".csv"))
    Z = Z[:,1]

else

    # this option should be used to load from stored .jld files

end

if dagp = "_dagp"
  burn_in = 1
end

Theta = data_res[:,1:N-2]' # stor data in column-major order
loglik = data_res[:,N-1]
accept_vec = data_res[:,N]

accept_rate = sum(accept_vec)/M
nbr_acf_lags =  50

acf = ones(N-2,nbr_acf_lags+1)

lower_q_int_limit = 2.5
upper_q_int_limit = 97.5

# use L"$\log r$"

if N == 6
    title_vec_log = [ L"$\log \kappa$"; L"$\log \gamma$"; L"$\log c$"; L"$\log d$"]
    title_vec = [ L"$\kappa$"; L"Gamma"; "c"; "d"]
elseif N == 4
    title_vec_log = [ L"$\log c$"; L"$\log d$" ]
    title_vec = [ "c"; "d" ];
elseif N == 5
    title_vec_log = [ L"$\log A$";L"$\log c$"; L"$\log d$" ]
    title_vec = [ "A";"c"; "d" ]
elseif N == 8
    title_vec_log = [ L"$\log A$"; L"$\log c$"; L"$\log d$"; L"$\log p_1$"; L"$\log p_2$"; L"$\log sigma$"]
    title_vec = [  "A    "; "c    "; "d    "; "p_1  "; "p_1  "; "sigma"]
elseif N == 7
    title_vec_log = [ L"$\log \kappa$"; L"$\log Gamma$"; L"$\log c$"; L"$\log d$"; L"$\log sigma$"];
    title_vec = [ L"$\kappa$"; "Gamma"; "c"; "d"; "sigma"]
elseif N == 9
    title_vec_log = [ L"$\log \kappa$"; L"$\log \gamma$"; L"$\log c$"; L"$\log d$"; L"$\log p_1$"; L"$\log p_2$"; L"$\log \sigma$"]
    title_vec = [  L"$\kappa$"; L"$\gamma$"; L"$c$"; L"$d$"; L"$p_1$"; L"$p_1$"; L"$\sigma$"]
else
    title_vec_log = [ L"$\log \kappa$"; L"$\log Gamma$"; L"$\log A$"; L"$\log c$"; L"$\log d$"; L"$\log g$"; L"$\log p_1$"; L"$\log p_2$"; L"$\log sigma$"]
    title_vec = [ L"$\kappa$"; "Gamma"; "A"; "c"; "d"; "g"; "p_1"; "p_1"; "sigma"]
end

acceptance_rate = sum(accept_vec[burn_in:end])/length(accept_vec[burn_in:end])

# print info

@printf "Accept rate: %.4f %% \n" acceptance_rate*100

@printf "True parameter values:\n"
Base.showarray(STDOUT,theta_true,false)
@printf "\n"

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

#on log-scale
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    PyPlot.plot(Theta[i,:])
    PyPlot.plot(ones(size(Theta,2),1)*theta_true[i], "k")
    PyPlot.ylabel(title_vec_log[i],fontsize=text_size)
end
PyPlot.xlabel("Iteration")

# on non-log scale
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    PyPlot.plot(exp.(Theta[i,:]))
    PyPlot.plot(ones(size(Theta,2),1)*exp(theta_true[i]), "k")
    PyPlot.ylabel(title_vec[i],fontsize=text_size)
end
PyPlot.xlabel("Iteration")

# plot trace plots after burn in

# on log-scale
PyPlot.figure()
x_axis = burn_in+1:size(Theta,2)
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    PyPlot.plot(x_axis, Theta[i,burn_in+1:end])
    PyPlot.plot(x_axis, ones(length(x_axis),1)*theta_true[i], "k")
    PyPlot.ylabel(title_vec_log[i])
end
PyPlot.xlabel("Iteration")

# on non-log scale
PyPlot.figure()
x_axis = burn_in+1:size(Theta,2)
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    PyPlot.plot(x_axis, exp.(Theta[i,burn_in+1:end]))
    PyPlot.plot(x_axis, ones(length(x_axis),1)*exp(theta_true[i]), "k")
    PyPlot.ylabel(title_vec[i])
end
PyPlot.xlabel("Iteration")

# calc acf for each marginal chain
for i = 1:N-2
    acf[i,2:end] = autocor(Theta[i,burn_in:end],1:nbr_acf_lags)
end

# plot acf
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    PyPlot.plot(0:nbr_acf_lags, acf[i,:])
    PyPlot.ylabel(title_vec_log[i])
end
PyPlot.xlabel("Lag")

# plot posterior
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    h = kde(Theta[i,burn_in+1:end])
    PyPlot.plot(h.x,h.density, "b")

    PyPlot.plot((theta_true[i], theta_true[i]), (0, maximum(h.density)), "k")

    if data_prior_dist_type == "Uniform"
        error("Uniform priors are not not implemented.")
    elseif data_prior_dist_type == "Normal"
        x_grid = (data_prior_dist[i,1]-4*data_prior_dist[i,2]):0.001:(data_prior_dist[i,1]+4*data_prior_dist[i,2])
        PyPlot.plot(x_grid, pdf(Normal(data_prior_dist[i,1],data_prior_dist[i,2]), x_grid) ,"g")
    end

    PyPlot.ylabel(title_vec_log[i])
end

# plot posterior, non-log scale
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    h = kde(exp.(Theta[i,burn_in+1:end]))
    PyPlot.plot(h.x,h.density, "b")
    PyPlot.plot((exp(theta_true[i]), exp(theta_true[i])), (0, maximum(h.density)), "k")
    PyPlot.ylabel(title_vec[i])
end

# plot log-lik
PyPlot.figure()
PyPlot.plot(loglik)
PyPlot.ylabel(L"$\log-likelhood")
PyPlot.xlabel("Iteration")

# plot acceptance rate for each K:th iteration
k = 500

accept_vec_k = zeros(size(Theta,2)/k ,1)
intervals = zeros(length(accept_vec_k),2)

j = 1
for r = 2:size(Theta,2)
    if mod(r-1,k) == 0
        accept_vec_k[j] = sum(accept_vec[r-k:r-1])/( r-1 - (r-k) ) * 100
        intervals[j,:] = [r-k, r-1]
        j = j +1
    end
end

accept_vec_k[end] = sum(accept_vec[(length(accept_vec)-k+1):end])/( k ) * 100

intervals[Int(size(Theta,2)/k),:] = [length(accept_vec)-k+1, length(accept_vec)]

# the "names" for the intervals are not used
#=
intervals_vec = {"start"};

for j = 1:length(intervals)
    intervals_vec(j) = {strcat(num2str(intervals(j,1)), "-", num2str(intervals(j,2)))}; % I should probably use vertical concatunate here!
end
=#

PyPlot.figure()
PyPlot.bar(1:length(accept_vec_k),accept_vec_k)
PyPlot.ylabel("Acceptance rate")
PyPlot.xlabel("Iteration")

# plot data
PyPlot.figure()
PyPlot.plot(1:length(Z),Z)
PyPlot.xlabel("Index")

PyPlot.figure()
PyPlot.plt[:hist](Z,50)

# leave results folder
cd("..")


# save results to jld file

# this feature is not needed...
