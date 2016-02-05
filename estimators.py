import numpy as np
import pymbar

# these functions should have standardized output:
# - how should I compare a variance estimate to a stochastic upper / lower bound?

def MBAR_from_AIS_paths(forward_paths, reverse_paths, reduced_potential_funcs):
   '''
   You have a list of k distributions (induced by the list of reduced_potential_funcs),
   and you are interested in the free energy difference between the first and last distribution.

   You've collected forward AIS samples
   (forward_paths, each starting in distribution 1 and ending in distribution k),
   and reverse AIS samples
   (reverse_paths, each starting in distribution k and ending in distribution 1). 

   Parameters
   ----------
   forward_paths : list of length-k lists of samples
      list of paths, where each path is a length-k list of samples

   reverse_paths : list of length-k lists of samples
      list of paths, where each path is a length-k list of samples

   reduced_potential_funcs : length-k list of functions
      each function accepts a sample and returns a real number

   Returns
   -------
   estimated_free_energy_difference : float
      estimate of free energy difference between first and last state on paths
   
   estimated_variance_in_free_energy_estimate : float
      estimated variance of the above estimate
  
   Notes
   -----
   - reduced_potential_func = minus the log unnormalized probability density
     (p(x) = e^[-u(x)] where u is the reduced_potential_func)

   '''
   
   # check that things are the correct shape
   n_forward = len(forward_paths)
   n_reverse = len(reverse_paths)
   k = len(forward_paths[0])
   # all forward and reverse paths the same length
   assert(len(set([len(path) for path in forward_paths])) == 1)
   assert(len(set([len(path) for path in reverse_paths])) == 1)
   assert(len(reverse_paths[0])==k)
   
   # form N_k (all states should have the same number of samples)
   N_paths = n_forward + n_reverse
   N_k = np.ones(k)*N_paths
   N = k*N_paths

   # form u_kn
   u_kn = np.zeros((k,N))

   reversed_reverse_paths = np.array([path[::-1] for path in reverse_paths])
   samples = np.hstack([forward_paths.flatten(),reversed_reverse_paths.flatten()])
   print(samples.shape)

   for i in range(k):
      u = reduced_potential_funcs[i]
      for j in range(N):
         u_kn[i,j] = u(samples[j])

   # run MBAR
   mbar = pymbar.MBAR(u_kn,N_k)
   return mbar
   # return free energy estimates + variance estimates
   #raise NotImplementedError

def bidirectional_mc_bounds(forward_paths,reverse_paths,reduced_potential_funcs):
   # compute forward log importance weights
   
   # double check signs here! I think these may both be sign-flipped
   f_log_imp_weights = [np.sum([reduced_potential_funcs[i](path[i]) for i in range(len(path))]) for path in forward_paths]
   r_log_imp_weights = [np.sum([reduced_potential_funcs[i](path[i]) for i in range(len(path))]) for path[::-1] in reverse_paths] 

   # construct estimates

   # return mean estimates + confidence intervals
   raise NotImplementedError
