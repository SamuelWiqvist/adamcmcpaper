# functions for the parallel particel filter including assiciated help functions

doc"""
    run_pf_paralell(Z::Array{Float64}, theta::Array{Float64}, theta_known::Array{Float64}, N::Int64, N_calc::Float64, dt::Float64, nbr_x0::Int64, nbr_x0_calc::Float64, nbr_x::Int64, nbr_x_calc::Float64, subsample_interval::Int64, subsample_interval_calc::Float64, print_on::Bool, store_weigths::Bool, Κ::Float64, Γ::Float64, A::Float64, B::Float64, c::Float64,d::Float64, f::Float64, g::Float64, power1::Float64, power2::Float64, b_const::Float64)

Runs each of the nbr_pf_proc estiamtions of log-likelihood for the parallel particel filter. Help funciton to pf_paralell.
"""
@fastmath function run_pf_paralell(Z::Array{Float64}, theta::Array{Float64},
  theta_known::Array{Float64}, N::Int64, N_calc::Float64, dt::Float64,dt_U::Float64, nbr_x0::Int64,
  nbr_x0_calc::Float64, nbr_x::Int64, nbr_x_calc::Float64, subsample_interval::Int64,
  subsample_interval_calc::Float64, print_on::Bool, return_weigths_and_particels::Bool,
  Κ::Float64, Γ::Float64, A::Float64, B::Float64, c::Float64,d::Float64, f::Float64,
  g::Float64, power1::Float64, power2::Float64, b_const::Float64)

  # set T
  T = length(Z)

  # Initilized wegits
  w = zeros(N,T)

  # Pre-allocate vector for storing particels
  x = zeros(N,T)

  # set delta for U process
  delta_i = dt_U

  # set start value for nbr_resample
  nbr_resample = 0

  # set start value for loglik
  loglik = 0.

  # pre-allocate vector for logweigths
  logweigths = zeros(N)

  # draw start values for particles
  xinit = 20 + (28-20)*rand(N)

  # pre-allocate vectors
  weigths = zeros(N)
  logweigths = zeros(N)
  dB_x0 = zeros(N,nbr_x0)
  dB_x = zeros(N, nbr_x)
  noise_x = zeros(N,nbr_x)

  # draw start noise
  noise_x0 = randn(N,nbr_x0)

  # set stat value
  xpred = xinit

  # propagate to the first value
  for i = 1:nbr_x0
    @simd for j = 1:N
      #dB_x0[j,i] = sqrt(dt*subsample_interval_calc/nbr_x_calc)*noise_x0[j,i]
      dB_x0[j,i] = sqrt(dt/nbr_x0_calc)*noise_x0[j,i]
    end
  end

  # propagate particels
  propagate_x_v2!(xpred, nbr_x0, nbr_x0_calc, dt, 1, 1., N, N_calc, dB_x0, A, B, c, d, f, g, power1, power2, b_const)


  # store particels
  x[:,1] = xpred

  # calc logweigths
  for j = 1:N
    logweigths[j] =  -log(Γ) - ((Z[1] - x[j,1])/Γ)^2/2 # no unnecessary normalizing constants
  end

  # find max
  constant_w = maximum(logweigths)

  # calc weigths
  for j = 1:N
    weigths[j] = exp(logweigths[j] - constant_w)
  end


  # update loglik
  loglik = loglik + constant_w + log(sum(weigths)) - log(N_calc) # p(Z_0 \mid theta)

  # calc normalized weigths
  w[:,1] = weigths/sum(weigths)


  # update loglik
  #loglik = loglik + log(sum(weigths.*w[:,1])) + sum(constant_w*w[:,1]);


  for t = 2:T

    # fill noise_x with new random numbers
    randn!(noise_x)

    # calc dB_x
    for i = 1:nbr_x
      for j = 1:N
        #dB_x[j,i] = sqrt(dt*subsample_interval_calc/nbr_x_calc)*noise_x[j,i]
        dB_x[j,i] = sqrt(dt/nbr_x_calc)*noise_x[j,i]
      end
    end

    #resample_cond = 1/sum(w[:,t-1].^2) < N/2
    # resample_cond = true # use ordinary boostrap filter
    # resample_cond = 1/sum(w[:,t-1].^2) < N/2  # use a condition for when to resample
    resample_cond = true
    # Only resample if ESS < N/2
    if resample_cond

      # calc resampled indecies
      ind = stratresample(w[:,t-1] , N)

      # resample particels
      x[:,t-1] = x[ind,t-1]

      # update nbr_resample
      nbr_resample = nbr_resample+1

      # reset weigths
      w[:,t-1] = 1/N


    end

    # set xpred
    xpred = x[:,t-1]

    # propagate particles
    propagate_x_v2!(xpred, nbr_x, nbr_x_calc, dt, subsample_interval, subsample_interval_calc, N,N_calc, dB_x,A,B,c,d,f,g,power1,power2,b_const)
    #ccall( (:propagate_x , "./libprop.so"), Void, (Csize_t,Csize_t,Prt{Float64}, Int64,Float64,Float64,Int64,Float64, Int64,Float64, Prt{Float64},Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64),   (N,nbr_x, xpred,nbr_x,nbr_x_calc,dt,subsample_interval,subsample_interval_calc,N,N_calc,dB_x,A,B,c,d,f,g,power1,power2,b_const) )

    # store particels
    x[:,t] = xpred

    # calc logweigths
    for j = 1:N
      logweigths[j] = -log(Γ) - 0.5*log(1-exp(-2*Κ*delta_i))  - (  (Z[t] - xpred[j] - exp(-Κ*delta_i)*(Z[t-1] - x[j,t-1]))/(Γ*sqrt(1-exp(-2*Κ*delta_i)) ) )^2/2.
    end

    # find max
    constant_w = maximum(logweigths)

    # calc weigths
    for j = 1:N
      weigths[j] = w[j,t-1]*exp(logweigths[j] - constant_w) # this is needed since we do not resample every time
    end

    #=
    if ! resample_cond  # use a condition for when to resample

      # if we do not resample we have to "correct" the weigths with w[:,t]
      for j = 1:N
        weigths[j] = w[j,t-1]*weigths[j]
      end

    end
    =#


    # calc normalized weigths
    w[:,t] = weigths/sum(weigths)

    # update loglik
    loglik = loglik + constant_w + log(sum(weigths)) - log(N_calc) # p(Z_t \mid theta)

    #loglik = loglik + log(sum(weigths.*w[:,t])) + sum(constant_w*w[:,t]);


  end

  # calc ESS for the particels at the last time step
  w_end_vec = 1/sum(w[:,end].^2)

  if print_on
    # print ESS and number of resamples
    @printf "ESS: %.4f\n" mean(w_end_vec)
    @printf "Nbr resample: %d \n" round(nbr_resample)
  end

  if return_weigths_and_particels
    return loglik, w, x
  else
    return loglik
  end


end

@inbounds begin

doc"""
    propagate_x_v2!(xpred::Array{Float64}, nbr::Int64,nbr_calc::Float64,dt::Float64,subsample_interval::Int64,subsample_interval_calc::Float64, N::Int64,N_calc::Float64, dB::Array{Float64},A::Float64,B::Float64,c::Float64,d::Float64,f::Float64,g::Float64,power1::Float64,power2::Float64,b_const::Float64)

Propagates the particels one time step.


"""
# @fastmath @inbounds
@fastmath function propagate_x_v2!(xpred::Array{Float64}, nbr::Int64,nbr_calc::Float64,dt::Float64,subsample_interval::Int64,subsample_interval_calc::Float64, N::Int64,N_calc::Float64, dB::Array{Float64},A::Float64,B::Float64,c::Float64,d::Float64,f::Float64,g::Float64,power1::Float64,power2::Float64,b_const::Float64)
  for k = 1:nbr
     @simd for j = 1:N #@simd
      #xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt*subsample_interval_calc/nbr_calc + b_const*dB[j,k]
      xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt/nbr_calc + b_const*dB[j,k]
    end
  end
end


end


doc"""
    stratresample(p , N)

Stratified resampling for sequential Monte Carlo.

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
