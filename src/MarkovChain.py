'''
Created on 14.01.2013

@author: psinger
'''

from __future__ import division

#import PathSim
#import csv
from collections import defaultdict, OrderedDict
import random
import collections
import operator
#import scipy.sparse as sp
import numpy as np
import sys
import math
#import operator
#from scipy import stats
from scipy.special import gammaln
from scipy.sparse import csr_matrix, coo_matrix
#from scipy.special import gamma
#import copy
#from random import choice
import itertools
import copy
import tables as tb

RESET_STATE = "-1"

UNKNOWN_STATE = "1"

#we need this for k = 0
FAKE_ELEM = "-10"

#Prior
#PRIOR = 1.00

class MarkovChain():
    '''
    Class for fitting a Markov chain of arbitrary order
    '''

    def __init__(self, k=1, reverse=False, use_prior=False,  reset=True, prior=1., specific_prior = None, specific_prior_vocab = None, modus="mle"):
        '''
        Constructor
        modus = specifies the modus of the class, there are two possibilities: modus='mle' is focused on working with mle matrices representing probabilities
        modus = 'bayes' focuses on working with bayesian evidence and only works with plain transition counts
        reverse = revert the paths
        use_prior = flag if script should use a prior
        reset = flag for using generic reset state
        prior = prior
        specific_prior = dictionary of specific priors for specific term combinations
        '''
        self.k_ = k
        self.reset_ = reset
        
        self.state_count_initial_ = 0
        self.states_initial_ = []
        self.parameter_count_ = 0
        self.observation_count_ = 0
        
        self.paths_ = list()
        self.paths_test_ = list()
        
        #probabilities
        self.transition_dict_ = defaultdict(lambda : defaultdict(float))
        
        self.prediction_position_dict_ = dict()
        self.vocabulary_ = None
        self.state_distr_ = defaultdict(float)
        
        self.states_ = dict()
        self.states_reverse_ = dict()
        self.dtype_ = np.dtype(float)
        self.reverse_ = reverse
        self.modus_ = modus
        
        self.use_prior_ = use_prior
        self.prior_ = prior



        self.specific_prior_ = specific_prior
        self.specific_prior_vocab_ = specific_prior_vocab

        
        #print self.specific_prior_
        if self.specific_prior_ is not None and k != 1:
            raise Exception("Using specific priors with higher orders not yet implemented!")
        if self.specific_prior_ is not None and self.specific_prior_vocab_ is None:
            raise Exception("Can't work with a specific prior without vocabulary information!")
        if self.specific_prior_ is not None and self.modus_ != "bayes":
            raise Exception("Specific prior only works mit Bayes modus!")

        self.proba_from_unknown_ = 0
        self.proba_to_unknown_ = dict()

    def _dict_divider(self, d): 
        '''
        Internal function for dict divider and smoothing
        '''

        if self.use_prior_ == True:
            smoothing_divider = float(self.state_count_initial_ * self.prior_)
            print "smoothing divider: ", smoothing_divider
            self.proba_from_unknown_ = self.prior_ / smoothing_divider
            print "proba_from_unknown_: ", self.proba_from_unknown_
            
            for k, v in d.iteritems():
                s = float(sum(v.values()))
                #smoothing_divider = float(sum([round(x*self.alpha_)+self.prior_ for x in self.specific_prior_[k].values()]))
                #smoothing_divider += float((self.state_count_initial_ - len(self.specific_prior_[k].values())) * self.prior_)
                
                divider = s + smoothing_divider
                self.observation_count_ += divider
                for i, j in v.iteritems():
                    v[i] = (j + self.prior_) / divider
                self.proba_to_unknown_[k] = self.prior_ / divider
                #print "row sum: ", (float(sum(v.values())) + ((self.state_count_initial_ - len(v)) * self.proba_to_unknown_[k]))
        else:
            for k, v in d.iteritems():
                s = float(sum(v.values()))
                self.observation_count_ += s
                for i, j in v.iteritems():
                    v[i] = j / s
                    
                #print "row sum: ", float(sum(v.values()))
                
    def _dict_ranker(self, d):
        '''
        Apply ranks to a dict according to the values
        Averages ties
        '''
        my_d = collections.defaultdict(list)
        for key, val in d.items():
            my_d[val].append(key)
        
        ranked_key_dict = {} 
        n = v = 1
        for _, my_list in sorted(my_d.items(), reverse=True):
            #v = n + (len(my_list)-1)/2.
            v = n + len(my_list)-1
            for e in my_list:
                n += 1
                ranked_key_dict[e] = v
                
        #little hack for storing the other unobserved average ranks
        #this is wanted so that we do not have to calculate it all the time again
        #ranked_key_dict[FAKE_ELEM] = n + ((self.state_count_initial_-len(ranked_key_dict)-1)/2.)
        ranked_key_dict[FAKE_ELEM] = self.state_count_initial_

        return ranked_key_dict
        
    def prepare_data(self, paths):
        '''
        preparing data
        ALWAYS CALL FIRST
        '''
        states = set()
        if self.reset_:
            states.add(RESET_STATE)    
        
        for line in paths:
            for ele in line:
                states.add(ele)
                self.state_distr_[ele] += 1
                
        #print self.state_distr_
                
        self.states_initial_ = frozenset(states)
                
        sum_state_occ = sum(self.state_distr_.values())
        for k,v in self.state_distr_.iteritems():
            self.state_distr_[k] = float(v) / float(sum_state_occ)

        #self.state_count_ = math.pow(float(len(states)), self.k_)
        self.state_count_initial_ = float(len(states))
        self.parameter_count_ = pow(self.state_count_initial_, self.k_) * (self.state_count_initial_ - 1)
        print "initial state count", self.state_count_initial_
        #print self.states_initial_
        
    def fit(self, paths, ret=False):
        '''
        fitting the data and constructing MLE
        ret = flag for returning the transition matrix
        '''
        print "====================="
        print "K: ", self.k_
        print "prior: ", self.prior_
        
        for line in paths:
            if self.reset_:
                self.paths_.append(self.k_*[RESET_STATE] + [x for x in line] + [RESET_STATE])
            else:
                self.paths_.append([x for x in line])
        
        for path in self.paths_:
            i = 0
            for j in xrange(self.k_, len(path)):
                elemA = tuple(path[i:j])
                i += 1
                elemB = path[j]
                if self.k_ == 0:
                    self.transition_dict_[FAKE_ELEM][elemB] += 1
                else:
                    self.transition_dict_[elemA][elemB] += 1
        
        #print self.transition_dict_

        
        if self.modus_ == "mle":
            self._dict_divider(self.transition_dict_)

        if ret:
            return self.transition_dict_
            
        #sys.exit()


        
    def loglikelihood(self):
        '''
        Calculating the log likelihood of the fitted MLE
        '''

        if self.modus_ != "mle":
            raise Exception("Loglikelihood calculation does not work with modus='bayes'")

        likelihood = 0
        prop_counter = 0
        
        for path in self.paths_:
            i = 0
            for j in xrange(self.k_, len(path)):
                elemA = tuple(path[i:j])
                i += 1
                elemB = path[j]
                if self.k_ == 0:
                    prop = self.transition_dict_[FAKE_ELEM][elemB]
                else: 
                    prop = self.transition_dict_[elemA][elemB]
                likelihood += math.log(prop)
                prop_counter += 1
                            
        print "likelihood", likelihood
        print "prop_counter", prop_counter
        return likelihood


    #@profile
    def bayesian_evidence(self, empirical_prior = 0, wrong_prior = 0):
        '''
        Calculating the bayesian evidence of the fitted MLE
        empirical_prior and wrong_prior are just for testing
        please do not use them except for testing
        '''
        if self.modus_ != "bayes":
            raise Exception("Bayesian evidence does not work with modus='mle'")
        
        print "starting to do bayesian evidence calculation!!"
        
        evidence = 0

        counter = 0

        i = 0


        #print len(self.transition_dict_.keys())

        #only works for order 1 atm
        if self.reset_ == False:
            allkeys = frozenset(self.transition_dict_.keys())
            for s in self.states_initial_:
                if (s,) not in allkeys:
                    self.transition_dict_[(s,)] = {}

        tmp = 0

        for k,v in self.transition_dict_.iteritems():
            tmp += 1
            #if tmp % 100 == 0:
                #print tmp

            first_term_enum = 0
            first_term_denom = 0        
            second_term_enum = 0
            second_term_denom = 0

            #start with combining prior knowledge with real data

            if self.specific_prior_ is not None and k[0] != RESET_STATE:
                if isinstance(self.specific_prior_, csr_matrix):
                    cx = self.specific_prior_.getrow(self.specific_prior_vocab_[k[0]])
                elif isinstance(self.specific_prior_, tb.group.RootGroup):
                    row = self.specific_prior_vocab_[k[0]]
                    indptr_first = self.specific_prior_.indptr[row]
                    indptr_second = self.specific_prior_.indptr[row+1]
                    data = self.specific_prior_.data[indptr_first:indptr_second]
                    indices = self.specific_prior_.indices[indptr_first:indptr_second]
                    indptr = np.array([0,indices.shape[0]])
                    if self.reset_:
                        shape = (1, self.state_count_initial_-1)
                    else:
                        shape = (1, self.state_count_initial_)
                    cx = csr_matrix((data, indices, indptr), shape=shape)
                else:
                    raise Exception("wrong specific prior format")

            done = set()
            done_counter = 0

            # if rowmax_prior > 0:
            #     tmp_max = max(v.values())
            for x, c in v.iteritems():
                prior = self.prior_ #+ (tmp_max - c) * 1.

                if empirical_prior > 0:
                    prior += empirical_prior



              #  if self.empirical_prior_ > 0:
               #     prior += (tmp_max - c) * self.empirical_prior_

                if self.specific_prior_ is not None and k[0] != RESET_STATE and x != RESET_STATE:
                    #print k[0], x
                    #prior += self.specific_prior_[self.specific_prior_vocab_[k[0]], self.specific_prior_vocab_[x]]
                    idx = self.specific_prior_vocab_[x]
                    prior += cx[0, idx]

                    done.add(idx)
                    # if k[0] in self.specific_prior_:
                    #     if x in self.specific_prior_[k[0]]:
                    #         prior += self.specific_prior_[k[0]][x]

                #print prior

                cp = c + prior
                              
                first_term_enum += prior
                first_term_denom += gammaln(prior)
                
                second_term_enum += gammaln(cp)
                second_term_denom += cp
                
                done_counter += 1
                counter += prior

            done = frozenset(done)



            #now lets add all prior information for which we do NOT have real data
            if self.specific_prior_ is not None and k[0] != RESET_STATE:#

                #if k[0] in self.specific_prior_:
                 #   for c in [b+self.prior_ for a,b in self.specific_prior_[k[0]].iteritems() if a not in v.keys()] :

                #cx = coo_matrix(self.specific_prior_.getrow(self.specific_prior_vocab_[k[0]]))
                #cx = self.specific_prior_.getrow(self.specific_prior_vocab_[k[0]]).tocoo()
                cx = cx.tocoo()
                for i,j,c in itertools.izip(cx.row, cx.col, cx.data):
                    #print "(%d, %d), %s" % (i,j,v)

                    #if self.specific_prior_vocab_reverse_[j] not in v.keys():
                    if j not in done:
                        c += self.prior_
                        first_term_enum += c
                        first_term_denom += gammaln(c)

                        second_term_enum += gammaln(c)
                        second_term_denom += c

                        done_counter += 1

                        counter += c

            
            #finally, we also need to cover those cases where no prior and no real data is available
            non_trans_count = int(self.state_count_initial_ - done_counter)

            prior = self.prior_

            if wrong_prior > 0:
                prior += wrong_prior#(tmp_max) * rowmax_prior

            counter += prior * non_trans_count

            #maybe I can skip this
            first_term_enum += (prior * non_trans_count)
            
            first_term_denom += (gammaln(prior) * non_trans_count)

            second_term_enum += (gammaln(prior) * non_trans_count)
            second_term_denom += (prior * non_trans_count)

            #do the final calculation
            first_term_enum = gammaln(first_term_enum)
            first_term = first_term_enum - first_term_denom
            
            second_term_denom = gammaln(second_term_denom)
            second_term = second_term_enum - second_term_denom
            

            evidence += (first_term + second_term)

        #print "final: %.30f" %evidence
        print "evidence", evidence
        #print self.prior_, empirical_prior, wrong_prior
        print "pseudo counts: ", counter
        return evidence
    
    def predict_eval(self, test, eval="rank"):
        '''
        Evaluating via predicting sequencies using MLE
        eval = choice between several evaluation metrics, "rank" is a ranked based approach and "top" checks whether
                true state is in the top K ranks
        ''' 
        
        if self.modus_ != 'mle':
            raise Exception("Prediction only works with MLE mode!")
        
        if self.use_prior_ != True:
            raise Exception("Prediction only works with smoothing on!")

        if eval == "rank":
            for k,v in self.transition_dict_.iteritems():
                print v
                self.prediction_position_dict_[k] = self._dict_ranker(v)

        known_states = frozenset(self.transition_dict_.keys())
        
        for line in test:
            #if self.k
            self.paths_test_.append(self.k_*[RESET_STATE] + [x for x in line] + [RESET_STATE])

        topx = 5
        position = 0.
        counter = 0.
        print "clicks test", len(self.paths_test_)
        
        for path in self.paths_test_:
            i = 0
            for j in xrange(self.k_, len(path)):
                elem = tuple(path[i:j])
                i += 1
                true_elem = path[j]
                
                if self.k_ == 0:
                    if eval == "rank":
                        p = self.prediction_position_dict_[FAKE_ELEM].get(true_elem,
                                                                          self.prediction_position_dict_[FAKE_ELEM][
                                                                              FAKE_ELEM])
                    elif eval == "top":
                        row = self.transition_dict_[FAKE_ELEM]
                        items = row.items()
                        random.shuffle(items)
                        row = OrderedDict(items)
                        top = dict(sorted(row.iteritems(), key=operator.itemgetter(1), reverse=True)[:topx]).keys()
                        if true_elem in top:
                            p = 1
                        else:
                            p = 0
                else:
                    #We go from an unknown state to some other
                    #We come up with an uniform prob distribution
                    if elem not in known_states:
                        if eval == "rank":
                            #p = self.state_count_initial_ / 2.
                            p = self.state_count_initial_
                        elif eval == "top":
                            prob = topx / self.state_count_initial_
                            if random.uniform <= prob:
                                p = 1
                            else:
                                p = 0
                    #We go from a known/learned state to some other
                    else:
                        if eval == "rank":
                            p = self.prediction_position_dict_[elem].get(true_elem,
                                                                         self.prediction_position_dict_[elem][
                                                                             FAKE_ELEM])
                        elif eval == "top":
                            row = self.transition_dict_[elem]
                            items = row.items()
                            random.shuffle(items)
                            row = OrderedDict(items)
                            top = dict(sorted(row.iteritems(), key=operator.itemgetter(1), reverse=True)[:topx]).keys()
                            if true_elem in top:
                                p = 1
                            else:
                                p = 0

                position += p
                counter += 1
                

        average_pos = position / counter 
        #print "unknown elem counter", unknown_elem_counter       
        print "counter", counter
        print "average position", average_pos
        return average_pos
       

        
            

                    
        