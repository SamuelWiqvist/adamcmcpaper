# script to run the forward simulations

using PyPlot
using KernelDensity
using Distributions
using DataFrames

include(pwd()*"/DWPSDE model/generate_data.jl")
include(pwd()*"/DWPSDE model/set_up.jl")
include(pwd()*"/DWPSDE model/mcmc.jl")

remove_missing_values(x) = reshape(collect(skipmissing(x)),7,:)

problem = "real data"
problem = "sim data small problem"

algorithm = "MCWM"
#algorithm = "DA"
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

# set-up parmaters for real data or sim data
if problem == "real data"


    dt = 0.35
    dt_U = 1

    B = 1.
    A = 0.01
    A_sign = 1
    f = 0.
    g = 0.03
    theta_known =  [A;A_sign;B;f;g]


else

    dt = 0.035
    dt_U = 1

    B = 1.
    A = 0.0025
    A_sign = -1
    f = 0.
    g = 0.
    theta_known =  [A;A_sign;B;f;g]

end

problem

# load data
data_res = convert(Array,readtable("DWPSDE model/analyses/"*problem*"/output_res"*jobname*".csv"))

M, N = size(data_res)

data_param = convert(Array,readtable("DWPSDE model/analyses/"*problem*"/output_param"*jobname*".csv"))

theta_true = data_param[1:N-2]
burn_in = Int64(data_param[N-2+1])

Z = convert(Array,readtable("DWPSDE model/analyses/"*problem*"/data_used"*jobname*".csv"))
Z = Z[:,1]

if algorithm == "DA" || algorithm == "ADA"
  burn_in = 1
end

# compute paramter estimations
Theta = remove_missing_values(data_res[:,1:N-2]') # stor data in column-major order
Theta = Theta[:,burn_in+1:end]
posterior_param_est = round.(mean(Theta,2),2)

# set paramters for forward sim
N_simulations = 3
nbr_sim_steps = length(Z)
start_val = Z[1]

# high density region of posterior
dist_from_posterior_mean = zeros(size(Theta,2))

for i = 1:size(Theta,2)
    dist_from_posterior_mean[i] = norm(Theta[:,i]-posterior_param_est)
end

idx = find(x -> x < quantile(dist_from_posterior_mean, 0.25),dist_from_posterior_mean)
posterior_high_dens = Theta[:,idx]
posterior_dist = Categorical(1/size(posterior_high_dens,2)*ones(size(posterior_high_dens,2)))

# pre-allocate data martix
forward_sim = zeros(N_simulations, nbr_sim_steps+1)

# run forward simulations
for i = 1:N_simulations
    idx = rand(posterior_dist)
    theta = posterior_high_dens[:,idx]
    forward_sim[i,:] = generate_data(theta, theta_known, 1., dt, dt_U, nbr_sim_steps, start_val)[1]
end

# plot forward simulations
text_size = 25
label_size = 20

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.subplot(221)
PyPlot.plot(forward_sim[1,:])
PyPlot.subplot(222)
PyPlot.plot(forward_sim[2,:])
PyPlot.subplot(223)
PyPlot.plot(forward_sim[3,:])
PyPlot.subplot(224)
PyPlot.plot(Z, "k")
#PyPlot.xlabel("Index",fontsize=text_size)
ax[:tick_params]("both",labelsize = label_size)


PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.subplot(221)
PyPlot.plt[:hist](forward_sim[1,:],50)
PyPlot.subplot(222)
PyPlot.plt[:hist](forward_sim[2,:],50)
PyPlot.subplot(223)
PyPlot.plt[:hist](forward_sim[3,:],50)
PyPlot.subplot(224)
PyPlot.plt[:hist](Z,50, color = "k")
ax[:tick_params]("both",labelsize = label_size)
