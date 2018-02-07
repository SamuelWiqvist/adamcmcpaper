# functions for plotting the resutls


function plot()


  # load data
  data_res = readtable("output_res.csv")
  (M,N) = size(data_res)
  Theta = data_res[:,1:N-2]
  loglik = data_res[:,N-1]
  accept_vec = data_res[:,N]




  if N == 6
      title_vec_log = [ 'log Kappa'; 'log Gamma'; 'log c    '; 'log d    '];
      title_vec = [ 'Kappa'; 'Gamma'; 'c    '; 'd    '];
  elseif N == 4
      title_vec_log = [ 'log c'; 'log d' ];
      title_vec = [ 'c'; 'd' ];
  elseif N == 5
      title_vec_log = [ 'log A';'log c'; 'log d' ];
      title_vec = [ 'A';'c'; 'd' ];
  elseif N == 8
      title_vec_log = [ 'log A    '; 'log c    '; 'log d    '; 'log p_1  '; 'log p_1  '; 'log sigma'];
      title_vec = [  'A    '; 'c    '; 'd    '; 'p_1  '; 'p_1  '; 'sigma'];
  elseif N == 7
      title_vec_log = [ 'log \kappa'; 'log \gamma'; 'log c     '; 'log d     '; 'log \sigma'];
      title_vec = [ '\Kappa'; '\gamma'; 'c     '; 'd     '; '\sigma'];
  elseif N == 9
      title_vec_log = [ 'log \kappa'; 'log \gamma'; 'log c     '; 'log d     '; 'log p_1   '; 'log p_1   '; 'log \sigma'];
      title_vec = [  '\kappa'; '\gamma'; 'c     '; 'd     '; 'p_1   '; 'p_1   '; '\sigma'];
  else
      title_vec_log = [ 'log \kappa'; 'log \gamma'; 'log A     '; 'log c     '; 'log d     '; 'log g     '; 'log p_1   '; 'log p_1   '; 'log \sigma'];
      title_vec = [ '\kappa'; '\gamma'; 'A     '; 'c     '; 'd     '; 'g     '; 'p_1   '; 'p_1   '; '\sigma'];
  end



end
