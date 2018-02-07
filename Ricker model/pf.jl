
################################################################################
######          Particle filters and help functions for pf                  #####
################################################################################


using PyPlot

# loglikelihood estimation

doc"""
    pf(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64,plotflag::Bool=false)

pf runs the bootstrap particel filter for the Ricker model and computes an unbiased
estimation of the loglikelihood function.
"""
function pf(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64,
plotflag::Bool=false)

  # set parameter values
  r = exp(theta[1])
  phi = exp(theta[2])
  sigma = exp(theta[3])

  # set startvalue for loglik
  loglik = 0.

  # set length pf data
  T = length(y)

  # pre-allocate matriceis
  x = zeros(N,T) # particels
  w = zeros(N,T) # weigts


  # set start values
  xint = rand(Uniform(1,30),N,1)
  #xint = rand(Uniform(1,200),N,1)

  #x_anc[:,1] = xint # set anc particels for t = 1

  # set gaussian noise
  e = rand(Normal(0,sigma), N,T)

  for t = 1:T

    if t == 1 # first iteration

      # propagate particels
      x[:,1] = r*xint.*exp(-xint .+ e[:,1]);


      # calc weigths and update loglik
      (w[:,t], loglik) = calc_weigths(y[t],x[:,t],phi,loglik,N)

    else

      # resample particels
      ind = stratresample(w[:,t-1], N)
      x_resample = x[ind,t-1]

      # propagate particels
      x[:,t] = r*x_resample.*exp(-x_resample .+ e[:,t])

      # calc weigths and update loglik
      (w[:,t], loglik) = calc_weigths(y[t],x[:,t],phi,loglik,N)
    end

  end

  if plotflag # plot ESS at last iteration
    @printf "ESS: %.4f\n" 1/sum(w[:,end].^2)
  end

  # return loglik
  return loglik
end

# latent state estimation


doc"""
    smc_filter(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64)

smc_filter runs the boostrap filteras an computes a state estiamtion. Based on the
code  at [link](https://github.com/umbertopicchini/SAEM-ABC/blob/master/nonlinear-gaussian/SAEM-SMC/smc_filter.m)
"""
function smc_filter(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64, resample_threshold::Float64, printon::Bool=false, plotdiagnostics::Bool=false, plottingtimes::Vector=[1])

  # set parameter values
  r = exp(theta[1])
  phi = exp(theta[2])
  sigma = exp(theta[3])

  # set startvalue for loglik
  loglik = 0.

  # set length pf data
  T = length(y)

  # pre-allocate matriceis
  x_hat = zeros(N) # particels
  logweigths = zeros(N) # weigts
  x_resample = zeros(N)

  # set start values
  xint = rand(Uniform(1,30),N,1)

  genealogy_indeces = zeros(Int64, T-1, N)
  genealogy_states = zeros(T,N)

  resample_counter = one(Int64)

  # set gaussian noise
  e = rand(Normal(0,sigma), N,T)



  for t = 1:T

    if t == 1 # first iteration

      # propagate particels
      x_hat = r*xint.*exp(-xint .+ e[:,1]);

      logweigths =  y[t]*log(x_hat.*phi)  .- x_hat*phi # compute logweigths
      logweigths = logweigths - maximum(logweigths) # remove max value
      logweigths = logweigths-log(sum(exp(logweigths))) # normalize

      # compute ess
      ess = 1/sum(exp(logweigths).^2)

      if ess < resample_threshold && t < T
        resample_counter = resample_counter+1
        ind = stratresample(logweigths, N)
        logweigths = -log(N)*ones(N)
      else
        ind = collect(1:N)
      end

      x_resample = x_hat[ind]

      # store states and indecies
      genealogy_indeces[1,:] = ind
      genealogy_states[1,:] = x_resample

    else

      # propagate particels
      x_hat = r*x_resample.*exp(-x_resample .+ e[:,t]);

      # compute weigths
      logweigths = logweigths + y[t]*log(x_hat.*phi)  .- x_hat*phi # compute logweigths
      logweigths = logweigths - maximum(logweigths) # remove max value
      logweigths = logweigths-log(sum(exp(logweigths))) # normalize

      # compute ess
      ess = 1/sum(exp(logweigths).^2)

      if ess < resample_threshold && t < T
        resample_counter = resample_counter+1
        ind = stratresample(exp(logweigths), N)
        logweigths = -log(N)*ones(N)
      else
        ind = collect(1:N)
      end

      x_resample = x_hat[ind]

      # store states and indecies

      if t < T
        genealogy_indeces[t,:] = ind
      end

      genealogy_states[t,:] = x_resample

      if plotdiagnostics
        if t in plottingtimes
          PyPlot.figure()
          PyPlot.plt[:hist](x_hat, 10, color = "b", alpha = 0.4, normed=true)
          PyPlot.hold(true)
          PyPlot.plt[:hist](x_resample, 10, color = "r", alpha = 0.4, normed=true)
        end
      end
    end

  end

  if printon == true
    @printf "Resample threshold: %d\n"  resample_threshold
    @printf "Nbr resamples: %d\n"  resample_counter
    @printf "End ess: %f\n"  1/sum(exp(logweigths).^2)
  end

  if plotdiagnostics
    # reconstruct all paths
    paths = zeros(T,N)

    for i = 1:N
      index_selected = i

      paths[end,i] = genealogy_states[end,index_selected]

      for t = T-1:-1:1
        index = genealogy_indeces[t, index_selected]
        paths[t,i] = genealogy_states[t,index]
        index_selected = index
      end
    end

    for t in plottingtimes
      PyPlot.figure()
      PyPlot.plt[:hist](paths[t,:], 10, color = "b", alpha = 0.4, normed=true)
    end

    PyPlot.figure()
    PyPlot.plot(paths)

    PyPlot.figure()
    PyPlot.plot(paths[:,1])

    std_paths = std(paths,2)

    @printf "Total std latent process: %f\n"  sum(std_paths)


    # reconstruct observed process

    paths_y = similar(paths)

    for i = 1:N
      for t = 1:T
        paths_y[t,i] = rand(Poisson(phi*paths[t,i]))
      end
    end

    std_paths_y = std(paths_y,2)
    @printf "Total std obs. process: %f\n"  sum(std_paths_y)


    for t in plottingtimes
      PyPlot.figure()
      PyPlot.plt[:hist](paths_y[t,:], 10, color = "b", alpha = 0.4, normed=true)
    end

    PyPlot.figure()
    PyPlot.plot(paths_y)

    PyPlot.figure()
    PyPlot.plot(paths_y[:,1])

  end


  # reconstruct the path
  path = zeros(T)

  index_selected = stratresample(exp(logweigths), 1)[1]

  path[end] = genealogy_states[end,index_selected]

  for t = T-1:-1:1
    index = genealogy_indeces[t, index_selected]
    path[t] = genealogy_states[t,index]
    index_selected = index
  end

  return path
end


doc"""
    abcsmc_filter(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64, δ_abc::Float64)

abcsmc_filter runs the ABC-SMC filter an computes a state estiamtion. Based on the
code  at [link](https://github.com/umbertopicchini/SAEM-ABC/blob/master/nonlinear-gaussian/SAEM-ABC/abcsmc_filter.m)
"""
function abcsmc_filter(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64, δ_abc::Float64, resample_threshold::Float64,printon::Bool=false, plotdiagnostics::Bool=false, plottingtimes::Vector=[1])

    # set parameter values
    r = exp(theta[1])
    phi = exp(theta[2])
    sigma = exp(theta[3])

    # set startvalue for loglik
    loglik = 0.

    # set length pf data
    T = length(y)

    # pre-allocate matriceis
    x_hat = zeros(N) # particels
    yobs = similar(x_hat)
    w = zeros(N) # weigts
    logweigths = zeros(N)
    x_resample = zeros(N)
    y_resample = zeros(N)

    # set start values
    xint = rand(Uniform(1,30),N,1)

    genealogy_indeces = zeros(Int64, T-1, N)
    genealogy_states = zeros(T,N)

    # set gaussian noise
    e = rand(Normal(0,sigma), N,T)


    resample_counter = 0

    for t = 1:T

      if t == 1 # first iteration

        # propagate particels
        x_hat = r*xint.*exp(-xint .+ e[:,1]);

        for i = 1:N
          yobs[i] = rand(Poisson(phi*x_hat[i]))
          logweigths[i] = -(yobs[i]-y[t])^2/(2*δ_abc^2)
        end

        logweigths = logweigths - maximum(logweigths) # remove max value
        logweigths = logweigths-log(sum(exp(logweigths))) # normalize

        ess = 1/sum(exp(logweigths).^2)

        if ess < resample_threshold && t < T
          # resample particels
          resample_counter = resample_counter+1
          ind = stratresample(exp(logweigths), N)
          logweigths = -log(N)*ones(N)
        else
          ind = collect(1:N)
        end

        x_resample = x_hat[ind]
        y_resample = yobs[ind]

        # store states and indecies
        genealogy_indeces[1,:] = ind
        genealogy_states[1,:] = y_resample


      else

        # propagate particels
        x_hat = r*x_resample.*exp(-x_resample .+ e[:,t]);

        for i = 1:N
          yobs[i] = rand(Poisson(phi*x_hat[i]))
          logweigths[i] = logweigths[i]-(yobs[i]-y[t])^2/(2*δ_abc^2)
        end

        logweigths = logweigths - maximum(logweigths) # remove max value
        logweigths = logweigths-log(sum(exp(logweigths))) # normalize

        ess = 1/sum(exp(logweigths).^2)

        if ess < resample_threshold
          # resample particels
          resample_counter = resample_counter+1
          ind = stratresample(exp(logweigths), N)
          logweigths = -log(N)*ones(N)
        else
          ind = collect(1:N)
        end

        x_resample = x_hat[ind]
        y_resample = yobs[ind]

        # store states and indecies

        if t < T
          genealogy_indeces[t,:] = ind
        end

        genealogy_states[t,:] = y_resample

        if plotdiagnostics
          if t in plottingtimes
            PyPlot.figure()
            PyPlot.plt[:hist](yobs, 10, color = "b", alpha = 0.4, normed=true)
            PyPlot.hold(true)
            PyPlot.plt[:hist](y_resample, 10, color = "r", alpha = 0.4, normed=true)
          end
        end
      end

    end


    if printon == true
      @printf "Resample threshold: %d\n"  resample_threshold
      @printf "Nbr resamples: %d\n"  resample_counter
      @printf "End ess: %f\n"  1/sum(exp(logweigths).^2)
    end


    if plotdiagnostics
      # reconstruct all paths
      paths = zeros(T,N)

      for i = 1:N
        index_selected = i

        paths[end,i] = genealogy_states[end,index_selected]

        for t = T-1:-1:1
          index = genealogy_indeces[t, index_selected]
          paths[t,i] = genealogy_states[t,index]
          index_selected = index
        end
      end

      std_paths = std(paths,2)

      @printf "Total std latent process: %f\n"  sum(std_paths)


      for t in plottingtimes
        PyPlot.figure()
        PyPlot.plt[:hist](paths[t,:], 10, color = "b", alpha = 0.4, normed=true)
      end

      PyPlot.figure()
      PyPlot.plot(paths)

      PyPlot.figure()
      PyPlot.plot(paths[:,1])


    end

    path = zeros(T)

    index_selected = stratresample(exp(logweigths), 1)[1]

    path[end] = genealogy_states[end,index_selected]

    for t = T-1:-1:1
      index = genealogy_indeces[t, index_selected]
      path[t] = genealogy_states[t,index]
      index_selected = index
    end

    return path

end


# observed state estimation

# help functions for particle filters

doc"""
    calc_weigths(y::Array{Float64},x::Array{Float64},phi::Float64,loglik::Float64,N::Int64)

Calculates the weigths in the particel filter and the estiamtes the loglikelihood value.
"""
function calc_weigths(y::Float64,x::Array{Float64},phi::Float64,loglik::Float64,N::Int64)
  logweigths = y*log(x.*phi)  .- x*phi # compute logweigths
  constant = maximum(logweigths) # find largets wegith
  weigths = exp(logweigths - constant) # subtract largets weigth and compute weigths
  loglik =  loglik + constant + log(sum(weigths)) - log(N) # update loglik
  return weigths/sum(weigths), loglik
end


doc"""
    stratresample(p , N)

Stratified resampling.

Sample N times with repetitions from a distribution on [1:length(p)] with probabilities p. See [link](http://www.cornebise.com/julien/publis/isba2012-slides.pdf).
"""
function stratresample(p , N)

  p = p/sum(p)  # normalize, just in case...

  cdf_approx = cumsum(p)
  cdf_approx[end] = 1
  #I = zeros(N,1)
  indx = zeros(Int64, N)
  U = rand(N,1)
  U = U/N + (0:(N - 1))/N
  index = 1
  for k = 1:N
    while (U[k] > cdf_approx[index])
      index = index + 1
    end
    indx[k] = index
  end

  return indx

end
