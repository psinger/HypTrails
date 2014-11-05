from __future__ import division

'''
This script is a simple example to show
how to use a specific prior.
This is mainly used for the HypTrails approach
which can be fetched from https://github.com/psinger/HypTrails
'''

__author__ = 'psinger'

import numpy as np
from src.MarkovChain import MarkovChain
from scipy.sparse import csr_matrix

paths = []
with open("data/test_case_3") as f:
    for line in f:
        if line.strip() == "":
            continue
        line = line.strip().split(" ")
        print len(line)
        #print line
        paths.append(np.array(line))

#this is without a specific prior
evidences = {}
markov = MarkovChain(use_prior=True, reset = True, modus="bayes")
markov.prepare_data(paths)
markov.fit(paths)

evidence = markov.bayesian_evidence()

print evidence

del markov

#this is with a very simple specific prior
evidences = {}
#we only have states 0 and 1
specific_prior = csr_matrix(np.array([[10,3],[4,1]]))
#need a vocab for assigning indices of the specific prior to a vocabulary
vocab = dict({"0":0, "1":1})

markov = MarkovChain(use_prior=True, reset = True, specific_prior=specific_prior,specific_prior_vocab=vocab, modus="bayes")
markov.prepare_data(paths)
markov.fit(paths)

evidence = markov.bayesian_evidence()

print evidence

del markov

