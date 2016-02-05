import numpy as np
import numpy.random as npr


def annealed_importance_sampling(draw_exact_initial_sample,
                                 transition_kernels,
                                 annealing_distributions,
                                 n_samples=1000):
    '''


    Parameters
    ----------

    draw_exact_initial_sample : function
        Signature:
            Parameters
            ----------
            none

            Returns
            -------
            sample : (d,), array-like

    transition_kernels : length-T list of functions
        Each function's signature:
            Parameters 
            ---------- 
            old_sample : (d,), array-like
            
            Returns
            -------
            new_sample : (d,), array-like

        Each function can be any transition operator that preserves its corresponding annealing distribution

    annealing_distributions:
        length-T list of functions

        Each function's signature:
            Parameters
            ----------
            x : (d,), array-like

            Returns
            -------
            unnormalized_density : float

        annealing_distributions[0] is the initial (unnormalized) density
        annealing_distributions[-1] is the target (unnormalized) density

    n_samples : positive integer

    Returns
    -------

    estimated_Z_ratio : (n_samples,), numpy.ndarray

    xs : length-T list of (T,d)-shaped numpy.ndarrays
        list of annealing paths

    weights : (n_samples,), numpy.ndarray
        estimated importance weights, one per sample

    ratios : (n_samples,T-1), numpy.ndarray
        intermediate importance weights
    '''

    dim=len(draw_exact_initial_sample())
    T = len(annealing_distributions)
    weights = np.ones(n_samples,dtype=np.double)
    ratios = []

    xs = []
    for k in range(n_samples):
        x = np.zeros((T,dim))
        ratios_ = np.zeros(T-1,dtype=np.double)
        x[0] = draw_exact_initial_sample()

        for t in range(1,T):


            f_tminus1 = annealing_distributions[t-1](x[t-1])
            f_t = annealing_distributions[t](x[t-1])

            ratios_[t-1] = f_t/f_tminus1
            weights[k] *= ratios_[t-1]

            x[t] = transition_kernels[t](x[t-1],target_f=annealing_distributions[t])

        xs.append(x)
        ratios.append(ratios_)

    estimated_Z_ratio = (np.cumsum(weights)/np.arange(1,len(weights)+1))

    return estimated_Z_ratio, np.array(xs), weights, np.array(ratios)

