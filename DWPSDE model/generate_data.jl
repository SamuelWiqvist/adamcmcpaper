# function for generating data from the DWP-SDE model with the V_{extended} potential function

doc"""
    generate_data(theta, theta_known, scale_grid)

Generates data from the model Z_t = X_t + U_t where
dX_t = -V_extended'(X_t)dt + dWt and U_t follows an UO-process.

# Inputs
* `theta`:true values for the unknown parameters.
* `theta_known`: known parameters.
* `scale_grid`: scaling of data set.

# Outputs
* `Z_sim`: simualted Z process.
* `dt`: dt for X process.
* `diff_dt`: differeance between dt for X and U process.
"""
function generate_data(theta, theta_known, scale_grid)

  # set parameters, use this to move between difference parameter combinations easily
  (Κ, Γ, A,A_sign,B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known, length(theta))

  # grid for the X_t process
  length_of_data = 25000
  #scale_grid = 4 #1  # choose lenght of data
  T = 1*scale_grid
  N = convert(Int64, scale_grid*length_of_data)
  dt_hat = T/N
  #dt = dt_hat*(1/4)*10^3*3.5
  dt = .35
  diff_dt = 1 #convert(Int64, round(10^-2. / dt))
  x_grid = 0:dt_hat:T

  # grid for the U_t process
  N_U = convert(Int64, scale_grid*length_of_data)
  dt_U = 1
  #dt_U = dt*diff_dt
  u_grid = 0:T/N_U:T

  Z_0 = 24.5 # start in the lower state

  X_sim = zeros(N+1)
  U_sim = zeros(N_U+1)
  dB = zeros(N+1)

  X_sim[1] = Z_0
  U_sim[1] = 0

  for i = 1:length(dB)
    dB[i] = sqrt(dt)*randn()
  end

  j = 1 # index for U_sim
  b_const = sqrt(2*sigma^2 / 2)

  #  should also consider the cache memory!
  for i = 1:N # numerical integration of the X process
    X_sim[i+1] = X_sim[i] - (A*X_sim[i] - f + (power2*abs(abs(c - B*X_sim[i])^power1/2 - d + g*X_sim[i])^(power2 - 1)*sign(abs(c - B*X_sim[i])^power1/2 - d + g*X_sim[i])*(g - (B*power1*abs(c - B*X_sim[i])^(power1 - 1)*sign(c - B*X_sim[i]))/2))/2)*dt + b_const*dB[i]
    if  mod(i, diff_dt) == 0
      U_sim[j+1] = rand(Normal( U_sim[j]*exp(-Κ*dt_U), sqrt( Γ^2*( 1 - exp( -2*Κ*dt_U ) ) ) ),1)[1]
      j = j + 1
    end
  end

  X_thinned = X_sim[1:diff_dt:end]

  Z_sim = X_thinned + U_sim

  return Z_sim, dt, diff_dt, X_thinned

end



doc"""
    generate_data(theta, theta_known)

Generates data from the model Z_t = X_t + U_t where
dX_t = -V_extended'(X_t)dt + dWt and U_t follows an UO-process.

# Inputs
* `theta`:true values for the unknown parameters.
* `theta_known`: known parameters.

# Outputs
* `Z_sim`: simualted Z process.
* `dt`: dt for X process.
* `diff_dt`: differeance between dt for X and U process.
"""
function generate_data(theta, theta_known)

  # set parameters, use this to move between difference parameter combinations easily
  (Κ, Γ, A,A_sign,B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known, length(theta))

  A = A_sign*A

  thinning_sim = 100

  N = Int64(3.5e6/thinning_sim)
  dt = 0.0035*thinning_sim#20*1e-9
  dt_U = 1#20*1e-9
  diff_dt = 1

  X_sim = zeros(N+1)
  U_sim = zeros(N+1)

  X_sim[1] = 35
  U_sim[1] = 0

  dB = zeros(N+1)


  for i = 1:length(dB)
    dB[i] = sqrt(dt)*randn()
  end

  j = 1 # index for U_sim
  b_const = sqrt(2*sigma^2 / 2)

  #  should also consider the cache memory!
  for i = 1:N # numerical integration of the X process
    X_sim[i+1] = X_sim[i] - (A*X_sim[i] - f + (power2*abs(abs(c - B*X_sim[i])^power1/2 - d + g*X_sim[i])^(power2 - 1)*sign(abs(c - B*X_sim[i])^power1/2 - d + g*X_sim[i])*(g - (B*power1*abs(c - B*X_sim[i])^(power1 - 1)*sign(c - B*X_sim[i]))/2))/2)*dt + b_const*dB[i]
    if  mod(i, diff_dt) == 0
      U_sim[j+1] = rand(Normal( U_sim[j]*exp(-Κ*dt_U), sqrt( Γ^2*( 1 - exp( -2*Κ*dt_U ) ) ) ),1)[1]
      j = j + 1
    end
  end

  X_thinned = X_sim[1:diff_dt:end]
  Z_sim = X_thinned + U_sim

  return Z_sim, X_thinned

end
