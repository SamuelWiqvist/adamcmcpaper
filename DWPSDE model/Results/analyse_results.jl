# Analyse results, used to print posterior inference results  

#using Plots
using PyPlot
using KernelDensity
using Distributions

using DataFrames

include(pwd()*"/utilities/posteriorinference.jl")

remove_missing_values(x) = reshape(collect(skipmissing(x)),7,:)

problem = "real data"
problem = "sim data small problem"

algorithm = "MCWM"
algorithm = "DA"
algorithm = "ADA"


if problem == "real data" && algorithm == "MCWM"
    jobname = "gp_training_7_par_lunarc_new_data_4_coresnew_data"
elseif problem == "real data" && algorithm == "DA"
    jobname = "_dagpest7new_datada_gp_mcmc"
elseif problem == "real data" && algorithm == "ADA"
    jobname = "_dagpest7new_dataada_gp_mcmc_dt"
elseif problem == "sim data small problem" && algorithm == "MCWM"
    jobname = "gp_training_7_par_lunarc_simdata_4_coressimdata"
elseif problem == "sim data small problem" && algorithm == "DA"
    jobname = "_dagpest7simdatada_gp_mcmc"
else #problem == "sim data scaled up problem" && algorithm = "ADA"
    jobname = "_dagpest7simdataada_gp_mcmc_dt"
end

if problem == "real data"
    plot_theta_true = false
else
    plot_theta_true = true
end

data_res = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_res"*jobname*".csv"))

M, N = size(data_res)

data_param = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_param"*jobname*".csv"))

theta_true = data_param[1:N-2]
burn_in = Int64(data_param[N-2+1])

data_prior_dist = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_prior_dist"*jobname*".csv"))

data_prior_dist_type = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_prior_dist_type"*jobname*".csv"))
data_prior_dist_type = data_prior_dist_type[2]

Z = convert(Array,readtable("DWPSDE model/Results/"*problem*"/data_used"*jobname*".csv"))
Z = Z[:,1]

if algorithm == "DA" || algorithm == "ADA"
  burn_in = 1
end

Theta = remove_missing_values(data_res[:,1:N-2]') # stor data in column-major order
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
println("---------------------------------------------------")
print("Problem:")
print(problem)
println("")
print("Algorithm:")
print(algorithm)
println("")
println("")


@printf "Accept rate: %.4f %% \n" round(acceptance_rate*100,2)

println("")

@printf "True parameter values:\n"
Base.showarray(STDOUT,round.(theta_true,2),false)
@printf "\n"

println("")

@printf "Posterior mean:\n"
Base.showarray(STDOUT,round.(mean(Theta[:,burn_in+1:end],2),2),false)
@printf "\n"

println("")

@printf "Posterior standard deviation:\n"
Base.showarray(STDOUT,round.(std(Theta[:,burn_in+1:end],2),2),false)
@printf "\n"

println("")

@printf "Posterior quantile intervals (2.5th and 97.5th quantiles as default):\n"
Base.showarray(STDOUT,round.(calcquantileint(Theta[:,burn_in+1:end],lower_q_int_limit,upper_q_int_limit),2),false)
@printf "\n"
println("---------------------------------------------------")

# plot trace plots

text_size = 15
label_size = 10

#on log-scale
PyPlot.figure(figsize=(10,20))
ax = axes()
for i = 1:N-2
    ax1 = PyPlot.subplot(N-2,1,i)
    PyPlot.plot(Theta[i,:])
    plot_theta_true == true ? PyPlot.plot(ones(size(Theta,2),1)*theta_true[i], "k") : 2
    PyPlot.ylabel(title_vec_log[i],fontsize=text_size)
    i < 7 ? ax1[:axes][:get_xaxis]()[:set_ticks]([]) : 2
end
PyPlot.xlabel("Iteration",fontsize=text_size)
ax[:tick_params]("both",labelsize = label_size)

# on non-log scale
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    PyPlot.plot(exp.(Theta[i,:]))
    plot_theta_true == true ? PyPlot.plot(ones(size(Theta,2),1)*exp(theta_true[i]), "k") :
    PyPlot.ylabel(title_vec[i],fontsize=text_size)
end
PyPlot.xlabel("Iteration")

# plot trace plots after burn in

# on log-scale
PyPlot.figure(figsize=(10,20))
ax = axes()
x_axis = burn_in+1:size(Theta,2)
for i = 1:N-2
    ax1 = PyPlot.subplot(N-2,1,i)
    PyPlot.plot(x_axis, Theta[i,burn_in+1:end])
    plot_theta_true == true ? PyPlot.plot(x_axis, ones(length(x_axis),1)*theta_true[i], "k") : 2
    PyPlot.ylabel(title_vec_log[i],fontsize=text_size)
    i < 7 ? ax1[:axes][:get_xaxis]()[:set_ticks]([]) : 2
end
PyPlot.xlabel("Iteration",fontsize=text_size)
ax[:tick_params]("both",labelsize = label_size)

# on non-log scale
PyPlot.figure()
x_axis = burn_in+1:size(Theta,2)
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    PyPlot.plot(x_axis, exp.(Theta[i,burn_in+1:end]))
    plot_theta_true == true ? PyPlot.plot(x_axis, ones(length(x_axis),1)*exp(theta_true[i]), "k") :
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

    if data_prior_dist_type == "Uniform"
        error("Uniform priors are not not implemented.")
    elseif data_prior_dist_type == "Normal"
        x_grid = (data_prior_dist[i,1]-4*data_prior_dist[i,2]):0.001:(data_prior_dist[i,1]+4*data_prior_dist[i,2])
        PyPlot.plot(x_grid, pdf(Normal(data_prior_dist[i,1],data_prior_dist[i,2]), x_grid) ,"g")
    end

    plot_theta_true == true ? PyPlot.plot((theta_true[i], theta_true[i]), (0, maximum(h.density)), "k") :

    PyPlot.ylabel(title_vec_log[i])
end

# plot posterior, non-log scale
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    h = kde(exp.(Theta[i,burn_in+1:end]))
    PyPlot.plot(h.x,h.density, "b")
    plot_theta_true == true ? PyPlot.plot((exp(theta_true[i]), exp(theta_true[i])), (0, maximum(h.density)), "k") :
    PyPlot.ylabel(title_vec[i])
end

# plot log-lik
PyPlot.figure()
PyPlot.plot(loglik)
PyPlot.ylabel(L"$\log-likelhood")
PyPlot.xlabel("Iteration")
