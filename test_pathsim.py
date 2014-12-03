'''
Created on 30.05.2014

@author: psinger
'''

from src.PathSim import PathSim
import numpy as np

for window_size in [2, 3]:
    print "==========="
    print window_size
    print "==========="

    sim = PathSim(window_size=window_size, sim_func="cosine", delimiter=" ")
    
    sim.fit("data/test_case_1")

    print sim.sim("1","1")
    print sim.sim("1","3")


