function propagate_x_v2_s2sub!(xpred::SharedArray, index::UnitRange, k::Int64,nbr::Int64,nbr_calc::Float64,dt::Float64,subsample_interval::Int64,subsample_interval_calc::Float64, N::Int64,N_calc::Float64, dB::SharedArray,A::Float64,B::Float64,c::Float64,d::Float64,f::Float64,g::Float64,power1::Float64,power2::Float64,b_const::Float64)
    for j in index
      #xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt*subsample_interval_calc/nbr_calc + b_const*dB[j,k]
      xpred[j] = xpred[j] - (A*xpred[j] - f + (power2*abs(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])^(power2 - 1)*sign(abs(c - B*xpred[j])^power1/2 - d + g*xpred[j])*(g - (B*power1*abs(c - B*xpred[j])^(power1 - 1)*sign(c - B*xpred[j]))/2))/2)*dt/nbr_calc + b_const*dB[j,k]
    end
  end
end
