import numpy as np
import numpy.random as npr

class Model():
   def __init__(self,
                unnormalized_density,
                draw_sample,
                exact_integral=None,
                log_prob=None):
      self.unnormalized_density = unnormalized_density
      
      self.draw_sample = draw_sample
     
      if exact_integral != None:
         self.exact_integral = exact_integral
         self.tractable = True
      else:
         self.tractable = False
      if log_prob != None:
         self.log_prob = log_prob

def one_d_gaussian_factory(mean,variance):

   def log_prob(x):
      return -((x-mean)**2)/(2*variance)

   def f(x):
      return np.exp(log_prob(x))

   def draw_sample():
      return (npr.randn(1)*np.sqrt(variance))+mean

   exact_integral = np.sqrt(variance)*np.sqrt(2*np.pi)

   return Model(unnormalized_density=f,
                draw_sample = draw_sample,
                exact_integral = exact_integral,
                log_prob = log_prob)


def mvn_factory(mean_vector,covariance):
   inv_cov = np.linalg.inv(covariance)
   def f(x):
      return np.exp(-0.5*(x-mean_vector).dot(inv_cov).dot(x-mean_vector))

   d = len(mean_vector)
   exact_integral = (2*np.pi)**(-d/2.0) * np.linalg.det(covariance)**(-0.5)

   mvn = multivariate_normal(mean=mean_vector,cov=covariance)

   return Model(unnormalized_density = f,
                draw_sample = mvn.rvs,
                exact_integral = exact_integral)
                
      
