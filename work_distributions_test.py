import pymbar
import numpy as np
import matplotlib.pyplot as plt

def plot_forward_and_reverse_work(W_F,W_R):
   plt.plot(W_F,'.',label='Forward work')
   plt.plot(W_R,'.',label='Reverse work')
   plt.legend(loc='best')
   plt.xlabel('Sample')
   plt.ylabel('Work')
   plt.title('Forward and reverse work')

def plot_rmse(n_samples,estimated_DeltaFs,actual_DeltaF,label,error_thin):
   sq_error = np.sqrt((estimated_DeltaFs - actual_DeltaF)**2)
   rmse = sq_error.mean(0)
   d_rmse = sq_error.std(0)
   plt.errorbar(n_samples,rmse,yerr=d_rmse,errorevery=error_thin,label=label)
   plt.xlabel('# samples')
   plt.ylabel('RMSE')
   plt.title('Root mean squared error')

def plot_stdev(n_samples,estimated_DeltaFs,label):
   stdev = estimated_DeltaFs.std(0)
   plt.plot(n_samples,stdev,label=label)
   plt.xlabel('# samples')
   plt.ylabel('Stdev')
   plt.title('Stdev')

def plot_bias(n_samples,estimated_DeltaFs,actual_DeltaF,label,error_thin):
   error = estimated_DeltaFs-actual_DeltaF
   bias = error.mean(0)
   d_bias = error.std(0)
   plt.errorbar(n_samples,bias,yerr=d_bias,errorevery=error_thin,label=label)
   plt.xlabel('# samples')
   plt.ylabel('Bias')
   plt.title('Bias')

def plot_predicted_vs_actual_stdev(predicted,d_predicted,actual,label):
   plt.errorbar(predicted,actual,xerr=d_predicted,errorevery=1,label=label)
   plt.xlabel('Predicted stdev')
   plt.ylabel('Actual stdev')

   max_ = np.max(np.hstack([predicted,d_predicted]))
   line = np.linspace(0,max_)
   plt.plot(line,line,linestyle='--')
   plt.title('Predicted vs. actual stdev')

def gaussian_stress_test(name,
            DeltaF=10,
            sigma_F=10,
            n_samples=1000,
            n_replicates=100,
            min_samples=5,
            n_errbars=50
            ):

   # collect work samples
   actual_DeltaF = DeltaF

   W_Fs = []
   W_Rs = []
   for n in range(n_replicates):
      W_F,W_R = pymbar.testsystems.gaussian_work_example(n_samples,n_samples,mu_F=None,DeltaF=actual_DeltaF,sigma_F=sigma_F)
      W_Fs.append(W_F)
      W_Rs.append(W_R)

   return compare(W_Fs,W_Rs,actual_DeltaF,min_samples,n_errbars)

def AIS_stress_test(n_samples=1000,
               n_replicates=100,
               min_samples=5,
               n_errbars=50,
               name='AIS'):
   from ais import annealed_importance_sampling
   
   from models import one_d_gaussian_factory
   initial = one_d_gaussian_factory(0,1.0)
   target = one_d_gaussian_factory(3,0.1)
   actual_Delta_G = target.exact_integral / initial.exact_integral

   from annealing_distributions import GeometricMean
   n_annealing = 10
   annealing_schedule = np.linspace(0,1,n_annealing)

   from transition_kernels import gaussian_random_walk
   kernels = [gaussian_random_walk for _ in range(n_annealing)]


   from test_instance import TestInstance
   test_case = TestInstance(initial,target,kernels,
                     GeometricMean,annealing_schedule)
   annealing_distributions = test_case.annealing_distributions

   W_Fs,W_Rs = [],[]
   for i in range(n_replicates):
      _,_,weights_f,_ = annealed_importance_sampling(initial.draw_sample,
                                          kernels,
                                          annealing_distributions,
                                          n_samples)
      _,_,weights_r,_ = annealed_importance_sampling(target.draw_sample,
                                          kernels[::-1],
                                          annealing_distributions[::-1],
                                          n_samples)
      #W_Fs.append(np.log(weights_f))
      #W_Rs.append(np.log(weights_r))

      W_Fs.append(weights_f)
      W_Rs.append(weights_r)

   actual_DeltaF = np.log(target.exact_integral/initial.exact_integral)
   return compare(W_Fs,W_Rs,actual_DeltaF,min_samples,n_errbars,name)


def compare(W_Fs,W_Rs,actual_DeltaF,min_samples,n_errbars,name):
   n_replicates = len(W_Fs)
   n_samples = len(W_Fs[0])
   error_thin=int(n_samples/n_errbars)

   # compute BAR estimates
   DeltaFs,dDeltaFs=[],[]
   for n in range(n_replicates):
      W_F,W_R = W_Fs[n],W_Rs[n]
      DeltaF,dDeltaF = np.zeros(len(W_F)),np.zeros(len(W_F))
      for i in range(min_samples,len(W_F)):
         DeltaF[i],dDeltaF[i] = pymbar.BAR(W_F[:i],W_R[:i])
      DeltaFs.append(DeltaF[min_samples:])
      dDeltaFs.append(dDeltaF[min_samples:])
   DeltaFs = np.array(DeltaFs)
   dDeltaFs = np.array(dDeltaFs)

   n_samples = np.arange(len(DeltaFs.T))+min_samples

   # compute F/R AIS estimates
   bd_mc_mean,bd_mc_errors = [],[]

   for n in range(n_replicates):
      W_F,W_R = W_Fs[n],W_Rs[n]
      forward_estimates = np.log(np.cumsum(np.exp(W_F))[min_samples:]/np.arange(min_samples,len(W_F)))
      reverse_estimates = np.log(np.arange(min_samples,len(W_R))/np.cumsum(np.exp(W_R))[min_samples:])

      # mean of forward and reverse estimates
      bd_mc_estimates=(forward_estimates+reverse_estimates)/2

      # not sure what to put here: half the bound width?
      bd_mc_error = np.abs(forward_estimates-reverse_estimates)/2

      bd_mc_mean.append(bd_mc_estimates)
      bd_mc_errors.append(bd_mc_error)

   bd_mc_mean = np.array(bd_mc_mean)
   bd_mc_errors = np.array(bd_mc_errors)

   ### Plot things

   # F/R work
   plt.figure()
   plot_forward_and_reverse_work(W_F,W_R)
   plt.savefig('{0}_fr_work.jpg'.format(name),dpi=300)
   plt.close()

   # RMSE
   plt.figure()
   plot_rmse(n_samples,DeltaFs,actual_DeltaF,label='BAR',error_thin=error_thin)
   plot_rmse(n_samples,bd_mc_mean,actual_DeltaF,label='BDMC',error_thin=error_thin)
   plt.legend(loc='best')
   plt.savefig('{0}_rmse.jpg'.format(name),dpi=300)
   plt.close()

   # stdev
   plt.figure()
   plot_stdev(n_samples,DeltaFs,label='BAR')
   plot_stdev(n_samples,bd_mc_mean,label='BDMC')
   plt.legend(loc='best')
   plt.savefig('{0}_stdev.jpg'.format(name),dpi=300)
   plt.close()

   # bias
   plt.figure()
   plot_bias(n_samples,DeltaFs,actual_DeltaF,label='BAR',error_thin=error_thin)
   plot_bias(n_samples,bd_mc_mean,actual_DeltaF,label='BDMC',error_thin=error_thin)
   plt.hlines(0,0,n_samples[-1],linestyles='--')
   plt.legend(loc='best')
   plt.savefig('{0}_bias.jpg'.format(name),dpi=300)
   plt.close()

   # error estimates
   plt.figure()
   plot_predicted_vs_actual_stdev(dDeltaFs.mean(0),dDeltaFs.std(0),DeltaFs.std(0),label='BAR')
   plot_predicted_vs_actual_stdev(bd_mc_errors.mean(0),bd_mc_errors.std(0),bd_mc_mean.std(0),label='BDMC')
   plt.legend(loc='best')
   plt.savefig('{0}_error.jpg'.format(name),dpi=300)
   plt.close()

   return DeltaFs,dDeltaFs,bd_mc_mean,bd_mc_errors


if __name__=='__main__':
   experiments = []

   n_samples=1000
   n_replicates=100
   min_samples=5



   for DeltaF in [0,0.1,1,10,100,1000]:
      for sigma_F in [0.01,0.1,1,10]:
         experiments.append({'DeltaF':DeltaF,'sigma_F':sigma_F})

   results = []
   for experiment in experiments:
      name='df={0},sigf={1}'.format(experiment['DeltaF'],experiment['sigma_F'])
      result = gaussian_stress_test(
                    name=name,
                    n_samples=n_samples,
                    n_replicates=n_replicates,
                    min_samples=min_samples,
                    **experiment)
      results.append((experiment,result))


   # save results
   import cPickle
   f = open('results.pickle','w')
   cPickle.dump(results,f)
   f.close()
