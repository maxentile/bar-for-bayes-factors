from ais import annealed_importance_sampling

class TestInstance():
   def __init__(self,
                initial_model,
                target_model,
                transition_kernels,
                annealing_mode,
                annealing_schedule):
      ''' 
      Parameters
      ----------
      initial_model : Model

      target_model : Model

      transition_kernels : length-T list of functions

      annealing_mode : instance of AnnealingDistribution

      annealing_schedule : (T,), array-like

      '''

      self.initial_model = initial_model
      self.target_model = target_model
      self.transition_kernels = transition_kernels
      self.annealing_schedule = annealing_schedule
      self.annealing_mode = annealing_mode

      # construct annealing distributions
      density_0 = self.initial_model.unnormalized_density
      density_1 = self.target_model.unnormalized_density
      self.annealing_distributions = [self.annealing_mode(density_0,density_1,beta) for beta in self.annealing_schedule]

   def collect_forward_samples(self,n_samples=1000):
      z_f,xs_f,weights_f,ratios_f = annealed_importance_sampling(
					self.initial_model.draw_sample,
					self.transition_kernels,
					self.annealing_distributions,
					n_samples)
      self.xs_f = xs_f
      self.z_f,self.weights_f,self.ratios_f = z_f,weights_f,ratios_f
      return xs_f

   def collect_reverse_samples(self,n_samples=1000):
      z_r,xs_r,weights_r,ratios_r = annealed_importance_sampling(
                                        self.target_model.draw_sample,
                                        self.transition_kernels[::-1],
                                        self.annealing_distributions[::-1],
                                        n_samples)
      self.z_r,self.weights_r,self.ratios_r = z_r,weights_r,ratios_r
      self.xs_r = xs_r
      return xs_r
