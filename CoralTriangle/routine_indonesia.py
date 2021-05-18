import numpy as np
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import time
import random

#! The following two lines are different among regions:
from functions_deterministic import *
import parameters_indonesia as P

#! This function submits multiple "sub_jobs" to the cluster:
def bound(njob, job_i, array_size):
    """ Returns the range of values that the job job_i out of njob jobs
    should process.

    Args:
      njob: total number of jobs in the array.
      job_i: index of the current job.
      array_size: the size of the array to split.

    Returns:
      start: the index of array to start.
      stop: the index of the array to stop.
      Note that array[start:stop] returns array[start]...array[stop-1]; that is,
      array[stop] is excluded from the range.
      
    """
    step = np.ceil(float(array_size)/float(njob))
    print 'njob=', njob, 'job_i= ', job_i, 'array_size= ', array_size, 'step= ', step
    start = int(job_i*step)
    stop = int(min((job_i+1)*step, array_size))
    print 'start= ', start, 'stop= ', stop
    return start, stop


if __name__ == '__main__':
    # get job_i from the command line argument
    arg = sys.argv
    njob = int(arg[1])
    job_i  = int(arg[2])
    
    #SET NUMBER OF ITERATIONS

    seed_global = np.arange(0,P.iterations)
    start, stop = bound(njob, job_i, seed_global.size)
    seed_values = seed_global[start:stop] 
    
    for seed in seed_values:
        np.random.seed(seed)
        
       #The following line runs the simulation:
        
        N_0, Z_0, YEAR_0, N_1, Z_1, YEAR_1, N_2, Z_2, YEAR_2, mpa_status = run_full_sim(P)
        #N_0, Z_0, YEAR_0 = run_hindcast(P)
        
        (np.save("./output/N_hindcast_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.hindcast_label +"_"+ str(seed) + ".npy", N_0))
                
        (np.save("./output/Z_hindcast_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.hindcast_label +"_"+ str(seed) + ".npy", Z_0))        
       
        (np.save("./output/year_hindcast_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.hindcast_label +"_"+ str(seed) + ".npy", YEAR_0))   

        (np.save("./output/N_forecast1_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.temp_scenario1 +"_"+ str(seed) + ".npy", N_1))
                
        (np.save("./output/Z_forecast1_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.temp_scenario1 +"_"+ str(seed) + ".npy", Z_1))        
                
        (np.save("./output/year_forecast1_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.temp_scenario1 +"_"+ str(seed) + ".npy", YEAR_1)) 

        (np.save("./output/N_forecast2_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.temp_scenario2 +"_"+ str(seed) + ".npy", N_2))
                
        (np.save("./output/Z_forecast2_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.temp_scenario2 +"_"+ str(seed) + ".npy", Z_2)) 

        (np.save("./output/year_forecast2_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.temp_scenario2 +"_"+ str(seed) + ".npy", YEAR_2))                       
        
        (np.save("./output/mpa_status_"+ P.region +"_beta_"+ str(P.beta[0,0]) +"_V_"+ str(P.V[0,0]) +
                "_mpa_"+str(P.reserve_strategy)+ str(P.reserve_fraction) +"_algmax_"+ str(P.algmort_max) +"_"+ 
                P.temp_scenario1 +"_"+ P.temp_scenario2 +"_"+ str(seed) + ".npy", mpa_status))