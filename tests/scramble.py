__author__ = 'singerpp'

from scipy.sparse import rand, lil_matrix, csr_matrix, vstack
import numpy as np
import scipy
import numpy as np
import sys
import random
from joblib import Parallel, delayed

from sklearn.preprocessing import normalize
def distr_chips(matrix, chips, matrix_sum_final = None, norm=True):
    '''
    Trial roulette method for eliciting Dirichlet priors from
    expressed hypothesis matrix.
    Note that only the informative part is done here.
    :param matrix: csr_matrix A_k expressing theory H_k
    :param chips: number of overall (whole matrix) chips C to distribute
    :param matrix_sum_final: the final sum of the input matrix, can be provided if matrix.sum() does not suffice
    :param norm: set False if matrix does not need to be normalized
    :return: Dirichlet pseudo clicks in the shape of a matrix
    '''

    #print "chips", chips

    chips = float(chips)

    if float(chips).is_integer() == False:
        raise Exception, "Only use C = |S|^2 * k"

    n = matrix.shape[1]

    nnz = matrix.nnz

    #if the matrix has 100% sparsity, we equally distribute the chips
    if nnz == 0:
        x = chips / n
        matrix[:] = int(x)
        rest = chips - (int(x) * n)
        if rest != 0.:
            eles = matrix.data.shape[0]
            idx = random.sample(range(eles),int(rest))
            # i_idx = [] #random.sample(range(matrix.shape[0]),int(rest))
            # j_idx = [] #random.sample(range(matrix.shape[1]),int(rest))
            # for l in xrange(int(rest)):
            #     i_idx.append(random.choice(range(matrix.shape[0])))
            #     j_idx.append(random.choice(range(matrix.shape[1])))
            # print len(i_idx)
            matrix.data[idx] += 1
        return matrix

    if norm:
        if matrix_sum_final is None:
            matrix_sum_final = matrix.sum()
        #it may make sense to do this in the outer scripts for memory reasons
        matrix = (matrix / matrix_sum_final) * chips
    else:
        matrix = matrix * chips

    floored = matrix.floor()

    rest_sum = int(chips - floored.sum())

    if rest_sum > 0:

        print "-----"
        print rest_sum

        #print "rest sum", rest_sum

        matrix = matrix - floored
        #print matrix.data.shape, floored.data.shape

        #as we can assume that the indices and states are already
        #in random order, we can also assume that ties are handled randomly here.
        #Better randomization might be appropriate though
        idx = matrix.data.argpartition(-rest_sum)[-rest_sum:]

        print matrix
        print idx

        i, j = matrix.nonzero()



        i_idx = i[idx]
        j_idx = j[idx]

        print i_idx, j_idx

        if len(i_idx) > 0:
            floored[i_idx, j_idx] += 1

    #print "final sum", floored.sum()

    #print matrix.data.shape, floored.data.shape

    #assert(matrix.data.shape == floored.data.shape)

    floored.eliminate_zeros()

    #print type(floored)

    del matrix

    #print "prior calc done"

    #print floored.nnz

    ##print floored

    return floored

def distr_chips_row(matrix, chips, n_jobs=-1, norm=True):
    '''
    Trial roulette method for eliciting Dirichlet priors from
    expressed hypothesis matrix.
    This function works row-based. Thus, each row will receive the given number of chips!!!
    :param matrix: csr_matrix A_k expressing theory H_k
    :param chips: number of (single row) chips C to distribute
    :param n_jobs: number of jobs, default -1
    :param norm: set False if matrix does not need to be normalized
    :return: Dirichlet pseudo clicks in the shape of a matrix
    '''

    r = Parallel(n_jobs=n_jobs)(delayed(distr_chips)(matrix[i,:],chips,norm=norm) for i in xrange(matrix.shape[0]))

    return scipy.sparse.vstack(r)

a = csr_matrix(np.array([[  0.,  70.,  30.],
 [ 70.,   0.,  50.],
 [ 30. , 50. ,  0.]]))

prior = distr_chips_row(a, 10, n_jobs=1)

print prior.toarray()