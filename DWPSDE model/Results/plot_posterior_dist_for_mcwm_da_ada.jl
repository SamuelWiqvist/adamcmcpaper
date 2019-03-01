# File used for plot the posterior dist for MCWM DA and ADA

using Plots
using PyPlot
using KernelDensity
using Distributions
using DataFrames

remove_missing_values(x) = reshape(collect(skipmissing(x)),7,:)

# load data for MCWM

problem = "real data"
problem = "sim data small problem"


if problem == "real data"

  jobname_mcwm = "gp_training_7_par_lunarc_new_data_4_coresnew_data"
  jobname_da = "_dagpest7new_datada_gp_mcmc"
  jobname_ada = "_dagpest7new_dataada_gp_mcmc_dt"

else# problem == "sim data small problem"

  jobname_mcwm = "gp_training_7_par_lunarc_simdata_4_coressimdata"
  jobname_da = "_dagpest7simdatada_gp_mcmc"
  jobname_ada = "_dagpest7simdataada_gp_mcmc_dt"

end

if problem == "real data"
    plot_theta_true = false
else
    plot_theta_true = true
end

data_res = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_res"*jobname_mcwm*".csv"))

M, N = size(data_res)

data_param = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_param"*jobname_mcwm*".csv"))

theta_true = data_param[1:N-2]
burn_in = Int64(data_param[N-2+1])

data_prior_dist = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_prior_dist"*jobname_mcwm*".csv"))

data_prior_dist_type = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_prior_dist_type"*jobname_mcwm*".csv"))
data_prior_dist_type = data_prior_dist_type[2]

Z = convert(Array,readtable("DWPSDE model/Results/"*problem*"/data_used"*jobname_mcwm*".csv"))
Z = Z[:,1]

Theta_mcwm = remove_missing_values(data_res[burn_in:end,1:N-2]') # stor data in column-major order

burn_in = 1

data_res = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_res"*jobname_da*".csv"))
M, N = size(data_res)
Theta_da = remove_missing_values(data_res[burn_in:end,1:N-2]') # stor data in column-major order

data_res = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_res"*jobname_ada*".csv"))
M, N = size(data_res)
Theta_ada = remove_missing_values(data_res[burn_in:end,1:N-2]') # stor data in column-major order


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


# text and lable size
text_size = 25
label_size = 20

(1:N-2)[1]



# plot posterior
#PyPlot.figure()
for i = (1:N-2) # (1:N-2)[j] to only plot figure j
    #PyPlot.subplot(N-2,1,i)
    PyPlot.figure(figsize=(10,10))
    ax = axes()
    h_mcwm = kde(Theta_mcwm[i,:])
    PyPlot.plot(h_mcwm.x,h_mcwm.density, "b")
    h_da = kde(Theta_da[i,:])
    PyPlot.plot(h_da.x,h_da.density, "r")
    h_ada = kde(Theta_ada[i,:])
    PyPlot.plot(h_ada.x,h_ada.density, "r--")

    if plot_theta_true
        PyPlot.plot((theta_true[i], theta_true[i]), (0, maximum([maximum(h_mcwm.density);maximum(h_da.density); maximum(h_ada.density)])), "k")
    end

    if data_prior_dist_type == "Uniform"
        error("Uniform priors are not not implemented.")
    elseif data_prior_dist_type == "Normal"
        #x_grid = (data_prior_dist[i,1]-3*data_prior_dist[i,2]):0.001:(data_prior_dist[i,1]+3*data_prior_dist[i,2])
        x_grid = (minimum(h_mcwm.x)-0.1*data_prior_dist[i,2]):0.001:(maximum(h_mcwm.x)+0.1*data_prior_dist[i,2])

        PyPlot.plot(x_grid, pdf(Normal(data_prior_dist[i,1],data_prior_dist[i,2]), x_grid) ,"g")
    end

    #PyPlot.xlabel(title_vec_log[i],fontsize=text_size)
    #PyPlot.ylabel(L"Density",fontsize=text_size)
    ax[:tick_params]("both",labelsize = label_size)

end


# plot data
PyPlot.figure(figsize=(12,10))
ax = axes()
PyPlot.plot(1:length(Z),Z)
#PyPlot.xlabel("Index")
ax[:tick_params]("both",labelsize = label_size)

# plot hist of data
PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plt[:hist](Z,50)
#PyPlot.ylabel("Freq.", fontsize=text_size)
ax[:tick_params]("both",labelsize = label_size)
