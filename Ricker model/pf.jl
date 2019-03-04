
################################################################################
######          Particle filter and help functions for pf                  #####
################################################################################

doc"""
    pf(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64,
plotflag::Bool=false, return_weigths_and_particles::Bool=false)

pf runs the bootstrap particel filter for the Ricker model.
"""
function pf(y::Array{Float64}, theta::Array{Float64},theta_known::Float64,N::Int64,
plotflag::Bool=false, return_weigths_and_particles::Bool=false)

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
  x_anc = zeros(N,T+1) # ancestral particels

  # set start values
  xint = rand(Uniform(1,30),N,1)
  #xint = rand(Uniform(1,200),N,1)

  x_anc[:,1] = xint # set anc particels for t = 1

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


    x_anc[:,t+1] = x_resample # store ancestral particels

    # propagate particels
    x[:,t] = r*x_resample.*exp(-x_resample .+ e[:,t])

    # calc weigths and update loglik
    (w[:,t], loglik) = calc_weigths(y[t],x[:,t],phi,loglik,N)
  end

  end

  if plotflag # plot ESS at last iteration
  @printf "ESS: %.4f\n" 1/sum(w[:,end].^2)
  end

  if return_weigths_and_particles
  # return loglik, weigths and particels
  loglik, w, x
  else
  # return loglik
  return loglik
  end
end


# help functions for particle filter

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
