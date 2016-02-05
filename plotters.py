import matplotlib.pyplot as plt
import numpy as np

# to-do: plot multiple traces on the same axes (e.g. same stats
#        but from different algorithms

def plot_traces_as_a_fxn_of_samples(replicates,y_label=None,title=None):
   for replicate in replicates:
      plt.plot(replicate)
   if y_label != None:
      plt.ylabel(y_label)
   if title != None:
      plt.title(title)

def plot_mean_stdev_as_a_fxn_of_samples(replicates,y_label=None,title=None):
   replicates = np.vstack(replicates)
   mean = replicates.mean(0)
   stdev = replicates.std(0)

   plt.plot(mean)
   plt.fillbetween(mean-stdev,mean+stdev,alpha=0.3)

   if y_label != None:
      plt.ylabel(y_label)
   if title != None:
      plt.title(title)
