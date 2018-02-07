# this script contains code to compute the approximate log-likelihood surface
# for the Ricker model as a function of two parameters keeping the third parameter
# fixed

# load functions
include("rickermodel.jl")

using Plots
using PyPlot
using StatPlots

# set up problem
problem = set_up_problem(ploton = false)

problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1] #Array(readtable("y.csv"))[:,1]

y = problem.data.y
theta_true = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
N = 1000  #problem.alg_param.N
Theta_parameters = problem.prior_dist.Theta_parameters

grid_size = 100

loglik_true = pf(y,theta_true,theta_known,N,false,false)


function findvalcorr(A::Matrix, val)

  M,N = size(A)
  for m = 1:M
    for n = 1:N
      if A[m,n] == val
        return (m,n)
      end
    end
  end

end



println("min -l(theta)")
println(minimum(-loglik_true))
println("minimizer (r,rho,sigma)")
println(theta_true)


# set rho constant

r_hat = linspace(Theta_parameters[1,1], Theta_parameters[1,2], grid_size)
rho = theta_true[2]
sigma_hat = linspace(Theta_parameters[3,1], Theta_parameters[3,2], grid_size)

loglik_m = zeros(grid_size, grid_size)

for i = 1:grid_size
  for j = 1:grid_size
    theta = [r_hat[i];rho;sigma_hat[j]]
    loglik_m[i,j] =  pf(y,theta,theta_known,N,false,false)
    if isnan(loglik_m[i,j])
      loglik_m[i,j] = 0
    end
  end
end


maximum(-loglik_m)
findvalcorr(-loglik_m, maximum(-loglik_m))

minimum(-loglik_m)
findvalcorr(-loglik_m, minimum(-loglik_m))
(r_ind, sigma_ind) = findvalcorr(-loglik_m, maximum(-loglik_m))
(r,sigma) = (r_hat[r_ind], sigma_hat[sigma_ind])


println("Keeping rho fixed:")
println("min -l(theta):")
println(minimum(-loglik_m))
println("minimizer (r,sigma):")
println([r,sigma])



PyPlot.figure()
PyPlot.surf(r_hat, sigma_hat, -loglik_m)
PyPlot.xlabel("r")
PyPlot.ylabel("sigma")


# set signa constant

r_hat = linspace(Theta_parameters[1,1], Theta_parameters[1,2], grid_size)
rho_hat = linspace(Theta_parameters[2,1], Theta_parameters[2,2], grid_size)
sigma = theta_true[3]

loglik_m = zeros(grid_size, grid_size)

for i = 1:grid_size
  for j = 1:grid_size
    theta = [r_hat[i];rho_hat[j];sigma]
    loglik_m[i,j] =  pf(y,theta,theta_known,N,false,false)
    if isnan(loglik_m[i,j])
      loglik_m[i,j] = 0
    end
  end
end


maximum(-loglik_m)
findvalcorr(-loglik_m, maximum(-loglik_m))

minimum(-loglik_m)
findvalcorr(-loglik_m, minimum(-loglik_m))
(r_ind, rho_ind) = findvalcorr(-loglik_m, maximum(-loglik_m))
(r,rho) = (r_hat[r_ind], rho_hat[rho_ind])


println("Keeping sigma fixed:")
println("min -l(theta)")
println(minimum(-loglik_m))
println("minimizer (r,rho)")
println([r,rho])



PyPlot.figure()
PyPlot.surf(r_hat, rho_hat, -loglik_m)
PyPlot.xlabel("r")
PyPlot.ylabel("rho")


# set r constant

r = theta_true[1]
rho_hat = linspace(Theta_parameters[2,1], Theta_parameters[2,2], grid_size)
sigma_hat = linspace(Theta_parameters[3,1], Theta_parameters[3,2], grid_size)

loglik_m = zeros(grid_size, grid_size)

for i = 1:grid_size
  for j = 1:grid_size
    theta = [r;rho_hat[i];sigma_hat[j]]
    loglik_m[i,j] =  pf(y,theta,theta_known,N,false,false)
    if isnan(loglik_m[i,j])
      loglik_m[i,j] = 0
    end
  end
end



maximum(-loglik_m)
(r_ind, sigma_ind) = findvalcorr(-loglik_m, maximum(-loglik_m))


minimum(-loglik_m)
findvalcorr(-loglik_m, minimum(-loglik_m))
(rho_ind, sigma_ind) = findvalcorr(-loglik_m, maximum(-loglik_m))
(rho,sigma) = (rho_hat[rho_ind], sigma_hat[sigma_ind])


println("Keeping r fixed:")
println("min -l(theta)")
println(minimum(-loglik_m))
println("minimizer (rho,sigma)")
println([rho,sigma])



PyPlot.figure()
PyPlot.surf(rho_hat, sigma_hat, -loglik_m)
PyPlot.xlabel("rho")
PyPlot.ylabel("sigma")
