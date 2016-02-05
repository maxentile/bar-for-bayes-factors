# maybe Model should also have an optional instance method for drawing samples?

class Model():
   def __init__(self,likelihood_fxn,exact_integral=None):
      self.likelihood_fxn = likelihood_fxn

      if exact_integral != None:
         self.exact_integral = exact_integral
         self.tractable = True
      else:
         self.tractable = False


def one_d_gaussian_factory(mean,variance):
   def likelihood(x):
      return np.exp(-(x-mean)**2/(2 * variance))

   exact_integral = np.sqrt(variance)*np.sqrt(2*np.pi)

   return Model(likelihood,exact_integral)


