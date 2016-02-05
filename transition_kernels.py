import numpy.random as npr
import numpy as np

def gaussian_random_walk(x,
                 target_f,
                 n_steps=10,
                 scale=0.5):
   ''' 
   Random walk metropolis-hastings with spherical gaussian
   proposals.

   Parameters
   ----------
   x : (d,), array-like
      random-walk initial state
   target_f : callable
      target unnormalized density
   n_steps : int
   scale : float

   Returns
   -------
   x_prime : (d,), numpy.ndarray
   '''

   x_old = np.array(x)
   f_old = target_f(x_old)
   dim=len(x)

   for i in range(n_steps):

      proposal = x_old + npr.randn(dim)*scale
      f_prop = target_f(proposal)

      if (f_prop / f_old) > npr.rand():
         x_old = proposal
         f_old = f_prop

   x_prime = x_old
   return x_old

