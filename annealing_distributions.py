class AnnealingDistribution():
   ''' Accepts two unnormalized densities and an annealing parameter, 
   and returns a new unnormalized density '''

   def __init__(self,initial_density,target_density,beta):
      '''
      Parameters
      ----------
      initial_density : callable
         accepts one argument and returns a nonnegative number

      target_density : callable
         accepts one argument and returns a nonnegative number

      beta : float
         annealing parameter

      Returns
      -------
      annealed_density : callable
         accepts one argument and returns a nonnegative number

      '''
      pass

   def __call__(self,x):
      pass

class GeometricMean(AnnealingDistribution):
   ''''
   Annealed distribution is target(x)^beta initial(x)^(1-beta).

   Notes
   -----
   - When initial = prior = p(theta) and 
     target = posterior = p(theta) p(y | theta)
     then this is equivalent to "turning on" the likelihood
     by p(y | theta)^beta p(theta)

   '''

   def __init__(self,initial,target,beta):
      self.initial = initial
      self.target = target
      self.beta = beta

   def __call__(self,x):
      f1_x = self.initial(x)
      fT_x = self.target(x)
      return f1_x**(1-self.beta) * fT_x**self.beta
