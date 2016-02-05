import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
   # create gaussian test instance
   from models import one_d_gaussian_factory
   initial = one_d_gaussian_factory(0,1)
   target = one_d_gaussian_factory(1,0.1)
   actual_Delta_G = target.exact_integral / initial.exact_integral

   from annealing_distributions import GeometricMean
   n_annealing = 10
   annealing_schedule = np.linspace(0,1,n_annealing)

   from transition_kernels import gaussian_random_walk
   kernels = [gaussian_random_walk for _ in range(n_annealing)]


   from test_instance import TestInstance
   test_case = TestInstance(initial,target,kernels,
                            GeometricMean,annealing_schedule)

   # collect forward/reverse AIS paths
   n_forward = 100
   n_reverse = 100
   forward_paths = test_case.collect_forward_samples(n_forward)
   reverse_paths = test_case.collect_reverse_samples(n_reverse)

   # plot stochastic upper/lower bounds   

   # compute MBAR estimate + variance estimates
   from estimators import MBAR_from_AIS_paths,bidirectional_mc_bounds

   reduced_potential_funcs = []
   from annealing_distributions import LogGeometricMean

   class SignFlip():
      def __init__(self,f):
         self.f = f
      def __call__(self,x):
         return - self.f(x)

   for beta in test_case.annealing_schedule:
      reduced_potential_funcs.append(SignFlip(LogGeometricMean(initial.log_prob,target.log_prob,beta)))

   mbar = MBAR_from_AIS_paths(forward_paths,reverse_paths,reduced_potential_funcs)

   Deltaf_ij, dDeltaf_ij, Theta_ij = mbar.getFreeEnergyDifferences()
   Delta_G = Deltaf_ij[0,-1]
   dDelta_G = dDeltaf_ij[0,-1]
   print('Actual difference: {0:.3f}'.format(actual_Delta_G))
   print('MBAR estimated Delta_G: {0:.3f}'.format(Delta_G))
   print('MBAR estimated dDelta_G: {0:.3f}'.format(dDelta_G))
   print(Deltaf_ij)

   # to-do: bidirectional_mc_bounds


   # to-do: plot everything from evaluators.py
