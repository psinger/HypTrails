'''
Created on 09.09.2013

@author: psinger
'''

import numpy as np
from src.MarkovChain import MarkovChain
import src.MarkovTools as mt
import sys

#paths = np.array([[1,1,2,1,3], [3,3,1,1,2], [2,1,2,3,1,1,1,1,3]])
paths = []
with open("data/test_case_2") as f:
    for line in f:
        if line.strip() == "":
            continue
        line = line.strip().split(" ")
        print len(line)
        #print line
        paths.append(np.array(line))


max_model = 3

likelihoods = {}
parameters = {}
observations = {}
state_count_initial = {}


#this is for the MLE case
for i in range(0,max_model+1):
    markov = MarkovChain(k=i, use_prior=False, reset = False, modus="mle")
    markov.prepare_data(paths)
    markov.fit(paths)
    
    l = markov.loglikelihood()
    likelihoods[i] = l
    parameters[i] = markov.parameter_count_
    observations[i] = markov.observation_count_
    state_count_initial[i] = markov.state_count_initial_
    
    del markov

print likelihoods


#print some sample statistics (i.e., Akaike Information Criterion)
lrts, pvals, dfs = mt.likelihood_ratio_test(likelihoods, parameters)
print lrts
print dfs

#sys.exit()


aics = mt.akaike_information_criterion(lratios=lrts, dfs=dfs, null_model=max_model)
print aics

sys.exit()

evidences = {}

#this is for the Bayesian case
for i in range(0,max_model+1):
    markov = MarkovChain(k=i, use_prior=True, reset = True, modus="bayes")
    markov.prepare_data(paths)
    markov.fit(paths)

    evidence = markov.bayesian_evidence()
    evidences[i] = evidence

    del markov

#print some sample statistics (i.e., evidences)
for k,v in evidences.iteritems():
    print "k: ", k
    print "evidence", v
    print "param count", parameters[k]
    
model_probas = mt.bayesian_model_selection(evidences=evidences, params=parameters, penalty=False)

print model_probas
