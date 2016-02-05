import numpy as np
import numpy.random as npr

class Model():
   def __init__(self,
                unnormalized_density,
                draw_sample,
                exact_integral=None):
      self.unnormalized_density = unnormalized_density

      if exact_integral != None:
         self.exact_integral = exact_integral
         self.tractable = True
      else:
         self.tractable = False


def one_d_gaussian_factory(mean,variance):
   def f(x):
      return np.exp(-(x-mean)**2/(2 * variance))

   def draw_sample():
      return (npr.randn()*np.sqrt(variance))+mean

   exact_integral = np.sqrt(variance)*np.sqrt(2*np.pi)

   return Model(unnormalized_density=f,
                draw_sample = draw_sample,
                exact_integral=exact_integral)


