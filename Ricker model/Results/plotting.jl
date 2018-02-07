# This file contains the plotting function

# load packages
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


# plotting function


doc"""
    analyse_results(problem, res::Result)

Prints the results of the mcmc alg. and also plots the parameter posterior
and diagnostics plots.
"""
function analyse_results(Theta, loglik, accept_vec, prior_vec,
  theta_true, burn_in, Theta_parameters, lower_q_int_limit::Real=2.5,upper_q_int_limit::Real=97.5)

  # calc acceptance rate
  acceptance_rate = sum(accept_vec[burn_in:end])/length(accept_vec[burn_in:end])
  nbr_out_side_prior = sum(prior_vec)

  # calc acf
  acf_c1 = autocor(Theta[1,burn_in:end],1:50)
  acf_c2 = autocor(Theta[2,burn_in:end],1:50)
  acf_c3 = autocor(Theta[3,burn_in:end],1:50)

  # print results
  @printf "True value for r : %.4f, estimated value %.4f \n" exp(theta_true[1]) exp(mean(Theta[1,burn_in+1:end]))
  @printf "True value for phi: %.4f, estimated value %.4f \n" exp(theta_true[2]) exp(mean(Theta[2,burn_in+1:end]))
  @printf "True value for sigma: %.4f, estimated value %.4f \n" exp(theta_true[3]) exp(mean(Theta[3,burn_in+1:end]))

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


  # text and lable size
  text_size = 15
  label_size = 15


  # plot chains
  PyPlot.figure(figsize=(10,20))
  ax1 = PyPlot.subplot(311)
  PyPlot.plot(Theta[1,:])
  PyPlot.plot(ones(size(Theta,2),1)*theta_true[1], "k")
  PyPlot.ylabel(L"$\log r$",fontsize=text_size)
  ax1[:axes][:get_xaxis]()[:set_ticks]([])
  ax2 = PyPlot.subplot(312)
  PyPlot.plot(Theta[2,:])
  PyPlot.plot(ones(size(Theta,2),1)*theta_true[2], "k")
  PyPlot.ylabel(L"$\log \phi$",fontsize=text_size)
  ax2[:axes][:get_xaxis]()[:set_ticks]([])
  ax3 = PyPlot.subplot(313)
  PyPlot.plot(Theta[3,:])
  PyPlot.plot(ones(size(Theta,2),1)*theta_true[3], "k")
  PyPlot.ylabel(L"$\log \sigma$",fontsize=text_size)
  PyPlot.xlabel("Iteration",fontsize=text_size)


  # plot chains after burn in
  PyPlot.figure(figsize=(10,20))
  ax1 = PyPlot.subplot(311)
  PyPlot.plot(Theta[1,burn_in:end])
  PyPlot.plot(ones(size(Theta[:,burn_in:end],2),1)*theta_true[1], "k")
  PyPlot.ylabel(L"$\log r$",fontsize=text_size)
  ax1[:axes][:get_xaxis]()[:set_ticks]([])
  ax2 = PyPlot.subplot(312)
  PyPlot.plot(Theta[2,burn_in:end])
  PyPlot.plot(ones(size(Theta[:,burn_in:end],2),1)*theta_true[2], "k")
  PyPlot.ylabel(L"$\log \phi$",fontsize=text_size)
  ax2[:axes][:get_xaxis]()[:set_ticks]([])
  ax1 = PyPlot.subplot(313)
  PyPlot.plot(Theta[3,burn_in:end])
  PyPlot.plot(ones(size(Theta[:,burn_in:end],2),1)*theta_true[3], "k")
  PyPlot.ylabel(L"$\log \sigma$",fontsize=text_size)
  PyPlot.xlabel("Iteration",fontsize=text_size)


    # plot acf
  PyPlot.figure()
  PyPlot.subplot(311)
  PyPlot.plot(acf_c1,"--r")
  PyPlot.ylabel(L"log $r$")
  PyPlot.subplot(312)
  PyPlot.plot(acf_c2,"--r")
  PyPlot.ylabel(L"log $\phi$")
  PyPlot.subplot(313)
  PyPlot.plot(acf_c3,"--r")
  PyPlot.ylabel(L"log $\sigma$")

  # plot posterior dist

  # calc grid for prior dist

x_c1 = Theta_parameters[1,1]-0.5:0.01:Theta_parameters[1,2]+0.5
x_c2 = Theta_parameters[2,1]-0.5:0.01:Theta_parameters[2,2]+0.5
x_c3 = Theta_parameters[3,1]-0.5:0.01:Theta_parameters[3,2]+0.5

# calc grid for kernel dens est of marginal posterior
#x_r_kerneldens = linspace(minimum(Theta[1,burn_in:end]),maximum(Theta[1,burn_in:end]),500)
#x_phi_kerneldens = linspace(minimum(Theta[2,burn_in:end]),maximum(Theta[2,burn_in:end]),500)
x_c1_kerneldens = x_c1
x_c2_kerneldens = x_c2
x_c3_kerneldens = x_c3

# calc prior dist
priordens_c1 = pdf(Uniform(Theta_parameters[1,1], Theta_parameters[1,2]), x_c1)
priordens_c2 = pdf(Uniform(Theta_parameters[2,1], Theta_parameters[2,2]), x_c2)
priordens_c3 = pdf(Uniform(Theta_parameters[3,1], Theta_parameters[3,2]), x_c3)


h1 = kde(Theta[1,burn_in:end])
h2 = kde(Theta[2,burn_in:end])
h3 = kde(Theta[3,burn_in:end])

PyPlot.figure()
ax = axes()

subplot(311)
PyPlot.plot(h1.x,h1.density, "b")
PyPlot.plot(x_c1,priordens_c1, "g")
PyPlot.plot((theta_true[1], theta_true[1]), (0, maximum(h1.density)), "k")
PyPlot.ylabel(L"log $r$",fontsize=text_size)

subplot(312)
PyPlot.plot(h2.x,h2.density, "b")
PyPlot.plot(x_c2,priordens_c2, "g")
PyPlot.plot((theta_true[2], theta_true[2]), (0, maximum(h2.density)), "k")
PyPlot.ylabel(L"log $\phi$",fontsize=text_size)

subplot(313)
PyPlot.plot(h3.x,h3.density, "b")
PyPlot.plot(x_c3,priordens_c3, "g")
PyPlot.plot((theta_true[3], theta_true[3]), (0, maximum(h3.density)), "k")
PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)

# print loglik
PyPlot.figure()
PyPlot.plot(loglik)
PyPlot.ylabel("Loglik")

end
