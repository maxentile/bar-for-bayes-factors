

def rmse_curve(estimates,truth):
   '''
   Parameters
   ----------
   estimates : list of array-like
      each array-like in estimates is a sequence of numbers,
      approximating "truth"

   truth : float
      known value

   Returns
   -------
   rmse : array-like
      

   '''

   raise NotImplementedError

def bias_curve(estimates,truth):
   '''
   Parameters
   ----------
   estimates : list of array-like
      each array-like in estimates is a sequence of numbers, 
      approximating "truth"

   truth : float
      known value

   Returns
   -------
   bias_mean : array-like
   
   bias_var : array-like
   '''

   raise NotImplementedError


def stdev_curve(estimates,truth):
   '''
   Parameters
   ----------
   estimates : list of array-like
      each array-like in estimates is a sequence of numbers, 
      approximating "truth"

   truth : float
      known value

   Returns
   -------
   stdev_mean : array-like
      sample stdev
   stdev_var : array_like
      estimated variance in sample stdev
   '''

   raise NotImplementedError

def evaluate_error_estimates(estimated_variances, actual_variances):
   '''
   Parameters
   ----------
   estimated_variances : list of array-like
      each array-like in estimated_variances is a sequence of numbers, representing the variance estimate at a given number of samples

   actual_variances : list of array-like
      each array-like in actual_variances is a sequence of numbers, representing the sample variance at a given number of samples

   Returns
   -------
   mean : array-like
  
   var : array_like
 
   '''

   raise NotImplementedError
