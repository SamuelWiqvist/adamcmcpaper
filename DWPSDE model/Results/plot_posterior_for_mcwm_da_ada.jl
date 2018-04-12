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

load_data_from_files = true # load data from files or form some workspace

# load data for MCWM
dagp = false #  set to _dagp to load ER-GP file  o.w. use ""

dataset = "sim_data" # select simdata or realdata (i.e the new dataset)

if dataset == "sim_data"

  # results for sim data
  # important! These files should be update later on!!!
  #_dagpest7ada_gp_mcmc_dt
  #_dagpest7da_gp_mcmcMCWM
  # gp_training_7_par

  jobname_mcwm = "gp_training_7_par" # jobname for mcwm
  jobname_da = "_dagpest7da_gp_mcmcMCWM"
  jobname_ada = "_dagpest7ada_gp_mcmc_dt"

elseif dataset == "realdata"

  # select res for real data
  # important! These files should be update later on!!!
  jobname_mcwm = "mcwm_7_par_real_data_2" # jobname for mcwm
  jobname_da = "mcwm_7_par_real_data_2"
  jobname_ada = "mcwm_7_par_real_data_2"

end


if load_data_from_files

    data_res = convert(Array,readtable("output_res"*jobname_mcwm*".csv"))

    M, N = size(data_res)

    data_param = convert(Array,readtable("output_param"*jobname_mcwm*".csv"))

    theta_true = data_param[1:N-2]
    burn_in = Int64(data_param[N-2+1])

    data_prior_dist = convert(Array,readtable("output_prior_dist"*jobname_mcwm*".csv"))

    data_prior_dist_type = convert(Array,readtable("output_prior_dist_type"*jobname_mcwm*".csv"))
    data_prior_dist_type = data_prior_dist_type[2]

    Z = convert(Array,readtable("data_used"*jobname_mcwm*".csv"))
    Z = Z[:,1]

    Theta_mcwm = data_res[burn_in:end,1:N-2]' # stor data in column-major order

    burn_in = 1

    data_res = convert(Array,readtable("output_res"*jobname_da*".csv"))
    M, N = size(data_res)
    Theta_da = data_res[burn_in:end,1:N-2]' # stor data in column-major order

    data_res = convert(Array,readtable("output_res"*jobname_ada*".csv"))
    M, N = size(data_res)
    Theta_ada = data_res[burn_in:end,1:N-2]' # stor data in column-major order


else

    # this option should be used to load from stored .jld files

end


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
text_size = 15
label_size = 15

# plot posterior
#PyPlot.figure()
for i = 1:N-2
    #PyPlot.subplot(N-2,1,i)
    PyPlot.figure()
    h_mcwm = kde(Theta_mcwm[i,:])
    PyPlot.plot(h_mcwm.x,h_mcwm.density, "b")
    h_da = kde(Theta_da[i,:])
    PyPlot.plot(h_da.x,h_da.density, "r")
    h_ada = kde(Theta_ada[i,:])
    PyPlot.plot(h_ada.x,h_ada.density, "r--")

    PyPlot.plot((theta_true[i], theta_true[i]), (0, maximum([maximum(h_mcwm.density);maximum(h_da.density); maximum(h_ada.density)])), "k")

    if data_prior_dist_type == "Uniform"
        error("Uniform priors are not not implemented.")
    elseif data_prior_dist_type == "Normal"
        x_grid = (data_prior_dist[i,1]-4*data_prior_dist[i,2]):0.001:(data_prior_dist[i,1]+4*data_prior_dist[i,2])
        PyPlot.plot(x_grid, pdf(Normal(data_prior_dist[i,1],data_prior_dist[i,2]), x_grid) ,"g")
    end

    PyPlot.xlabel(title_vec_log[i],fontsize=text_size)
    PyPlot.ylabel(L"Density",fontsize=text_size)

end


# plot data
PyPlot.figure()
PyPlot.plot(1:length(Z),Z)
PyPlot.xlabel("Index")

#=
# plot posterior, non-log scale
PyPlot.figure()
for i = 1:N-2
    PyPlot.subplot(N-2,1,i)
    h = kde(exp.(Theta[i,burn_in+1:end]))
    PyPlot.plot(h.x,h.density, "b")
    PyPlot.plot((exp(theta_true[i]), exp(theta_true[i])), (0, maximum(h.density)), "k")
    PyPlot.ylabel(title_vec[i])
end
=#

# leave results folder
cd("..")
