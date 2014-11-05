from __future__ import division

__author__ = 'psinger'

#get it from https://github.com/psinger/PathTools
import PathTools as pt

from src.trial_roulette import *
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

'''
Simple test case to illustrate the HypTrails approach
'''

#get the trails
trails = []
with open("data/test_case_1") as f:
    for line in f:
        if line.strip() == "":
            continue
        line = line.strip().split(" ")
        print len(line)
        #print line
        trails.append(np.array(line))

#get distinct states
states = set()
for row in trails:
    col = list(row)
    for c in col:
        states.add(c)

#build the vocabulary for matrix A
vocab = dict(((t, i) for i, t in enumerate(states)))

#express hypothesis as matrix A
#just an arbitrary example here
A = lil_matrix((5,5))
#let us believe strongly in transitioning from state 1 to 2 and vice versa
A[vocab["1"],vocab["2"]] = 0.8
A[vocab["2"],vocab["1"]] = 0.8
#and a bit in 1 to 3 and vice versa
A[vocab["1"],vocab["3"]] = 0.4
A[vocab["3"],vocab["1"]] = 0.4

print vocab
print A

A = A.tocsr()

#number of chips C
#only informative part
chips = 25


prior = distr_chips(A, chips)

#prior=1. refers to the uniform part
markov = pt.MarkovChain(k=1, use_prior=True, reset = True, prior=1., specific_prior=prior,
                                    specific_prior_vocab = vocab, modus="bayes")
markov.prepare_data(trails)
markov.fit(trails)

evi1 = markov.bayesian_evidence()
print evi1

#let us check another hypothesis
#self-loop hypothesis
A = lil_matrix((5,5))
A.setdiag(1.)
A = A.tocsr()
prior = distr_chips(A, chips)

markov = pt.MarkovChain(k=1, use_prior=True, reset = True, prior=1., specific_prior=prior,
                                    specific_prior_vocab = vocab, modus="bayes")
markov.prepare_data(trails)
markov.fit(trails)

evi2 = markov.bayesian_evidence()
print evi2

#Bayes factor
B = evi1 - evi2
print B

#check http://en.wikipedia.org/wiki/Bayes_factor for interpretation
print "Decisive Bayes factor"
print "The first hypothesis is more plausible than the second (self-loop) hypothesis!"