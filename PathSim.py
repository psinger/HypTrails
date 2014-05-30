'''
Created on 07.01.2013

@author: Philipp Singer
'''

import sys
import csv
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import math
#from joblib import Memory


class PathSim():
    '''
    path class
    '''


    def __init__(self, window_size=2, delimiter=' ', sim_func='cosine'):
        '''
        Constructor
        '''      
        self.window_size_ = window_size
        self.delimiter_ = delimiter
        
        if sim_func not in ['cosine', 'mutual']:
            raise ValueError('similarity function not allowed')
        self.sim_func_ = sim_func
        
        self.coocs_ = None
        self.dtype_ = np.uint
        self.binary_ = None
        
    def _cooc_count_dicts_to_matrix(self, cooc_dict_dict):
        i_indices = []
        j_indices = []
        values = []
        vocabulary = self.vocabulary

        #print cooc_dict_dict
        for cooc_dict_key in cooc_dict_dict.keys():
            #print "key", cooc_dict_dict[cooc_dict_key]
            cooc_dict = cooc_dict_dict[cooc_dict_key]
            for term, count in cooc_dict.iteritems():
                i = vocabulary.get(cooc_dict_key)
                j = vocabulary.get(term)
                #print "j", j
                if i is not None and j is not None: 
                    i_indices.append(i)                   
                    j_indices.append(j)
                    values.append(count)
                    #symmetric
                    i_indices.append(j)                   
                    j_indices.append(i)
                    values.append(count)
            # free memory as we go
            cooc_dict.clear()

        #print len(i_indices)
        #print len(j_indices)
        #print len(values)

        shape = (max(vocabulary.itervalues()) + 1, max(vocabulary.itervalues()) + 1)
        #print shape
        #print len(values)
        spmatrix = sp.csr_matrix((values, (i_indices, j_indices)),
                                 shape=shape, dtype=self.dtype_)
        if self.binary_:
            spmatrix.data.fill(1)
        
        self.coocs_ = spmatrix
        #print spmatrix.shape
    
    def fit(self, paths, check_only = None):
        '''
        fitting the given paths
        paths = path file
        check_only = frozenset with elements you want to know
        '''
        with open(paths) as f:
            reader = csv.reader(f, delimiter=self.delimiter_)
            
            elements = set()
            cooc_dict = defaultdict(lambda : defaultdict(int))
            
            for line in reader:
                #print line
                #sys.exit()
                for i, v in enumerate(line):
                    
                    elements.add(v)
                    if self.window_size_ == "none":
                        max_it = len(line)
                    else:
                        max_it = self.window_size_
                    for j in xrange(1, max_it):
                        idx = i + j
                        if idx >= len(line):
                            break
                        elemA = v
                        elemB = line[idx]
                        if check_only is not None:
                            if elemA not in check_only and elemB not in check_only:
                                continue
                        if elemA == elemB:
                            #In this case they are equal, maybe break
                            continue
                        cooc_dict[elemA][elemB] += 1
                        #cooc_dict[elemB][elemA] += 1
        
        vocab = dict(((t, i) for i, t in enumerate(sorted(elements))))
        if not vocab:
            raise ValueError("no elements")
        self.vocabulary = vocab

        self._cooc_count_dicts_to_matrix(cooc_dict)
        
    def sim(self, elemA, elemB):
        '''
        determining similarity for a given pair
        '''
        keyA = self.vocabulary[elemA]
        keyB = self.vocabulary[elemB]

#        vecA = sp.csr_matrix(self.coocs_.getrow(keyA))
#        vecB = sp.csr_matrix(self.coocs_.getrow(keyB))

        vecA = self.coocs_.getrow(keyA)
        vecB = self.coocs_.getrow(keyB)

        sim_func = getattr(self, self.sim_func_)
        return sim_func(vecA, vecB)

    def cosine(self, u, v):
        '''
        cosine similarity
        '''
        len_u = math.sqrt(u.dot(u.T)[0,0])
        len_v = math.sqrt(v.dot(v.T)[0,0])
        
        cs = u.dot(v.T)[0,0] / (len_u * len_v)
        
        if cs >= 0:
            return cs
        else:
            return 0
        
    def mutual(self):
        print "mutual"