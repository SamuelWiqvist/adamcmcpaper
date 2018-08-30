# This file contains the plotting function

# load packages
using PyPlot
using KernelDensity
using Distributions


include(pwd()*"/utilities/posteriorinference.jl")


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
  Base.showarray(STDOUT,round(mean(Theta[:,burn_in+1:end],2),2),false)
  @printf "\n"

  @printf "Posterior standard deviation:\n"
  Base.showarray(STDOUT,std(Theta[:,burn_in+1:end],2),false)
  @printf "\n"

  @printf "Posterior quantile intervals (2.5th and 97.5th quantiles as default):\n"
  Base.showarray(STDOUT,round(calcquantileint(Theta[:,burn_in+1:end],lower_q_int_limit,upper_q_int_limit),2),false)
  @printf "\n"

  @printf "Loss for parameter estimations:\n"
  Base.showarray(STDOUT,loss(theta_true, Theta[:,burn_in+1:end]),false)
  @printf "\n"


  # text and lable size
  text_size = 15
  label_size = 10


  # plot chains
  PyPlot.figure(figsize=(10,20))
  ax = axes()
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
  ax[:tick_params]("both",labelsize = label_size)


  # plot chains after burn in
  PyPlot.figure(figsize=(10,20))
  ax = axes()
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
  ax[:tick_params]("both",labelsize = label_size)


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
priordens_c1 = pdf.(Uniform(Theta_parameters[1,1], Theta_parameters[1,2]), x_c1)
priordens_c2 = pdf.(Uniform(Theta_parameters[2,1], Theta_parameters[2,2]), x_c2)
priordens_c3 = pdf.(Uniform(Theta_parameters[3,1], Theta_parameters[3,2]), x_c3)


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


posterior_samples = Theta[:,burn_in:end]

PyPlot.figure()
counter = 1
for i = 1:3
  for j = 1:3
    PyPlot.subplot(3,3,counter) # set subplot
    if i == j
      # plot hist on diag. elements
      h = PyPlot.plt[:hist](posterior_samples[i,:],50)
      PyPlot.plot((theta_true[i], theta_true[i]), (0, maximum(h[1])), "k")
    else
      # plot scatters on non-diagonal elements
      PyPlot.scatter(posterior_samples[j,:], posterior_samples[i,:])
    end
    counter += 1

    if i == 1 && j == 1
      PyPlot.ylabel(L"log $r$")
    elseif i == 3 && j == 1
      PyPlot.xlabel(L"log $r$")
    end

    if j == 1 && i == 2
      PyPlot.ylabel(L"log $\phi$")
    elseif i == 3 && j == 2
      PyPlot.xlabel(L"log $\phi$")
    end

    if j == 1 && i == 3
      PyPlot.ylabel(L"log $\sigma$")
    elseif i == 3 && j == 3
      PyPlot.xlabel(L"log $\sigma$")
    end

  end
end


end
