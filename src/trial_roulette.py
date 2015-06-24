from __future__ import division

__author__ = 'psinger'

import tables as tb
import time
import scipy
import numpy as np
import sys
import random
from joblib import Parallel, delayed

def hdf5_save(matrix, filename, dtype=np.dtype(np.float64)):
    '''
    Helper function for storing scipy matrices as PyTables HDF5 matrices
    see http://www.philippsinger.info/?p=464 for further information
    :param matrix: matrix to store
    :param filename: filename for storage
    :param dtype: dtype
    :return: True
    '''

    #print matrix.shape

    atom = tb.Atom.from_dtype(dtype)

    f = tb.open_file(filename, 'w')

    #print "saving data"
    filters = tb.Filters(complevel=5, complib='blosc')
    out = f.create_carray(f.root, 'data', atom, shape=matrix.data.shape, filters=filters)
    out[:] = matrix.data

    #print "saving indices"
    out = f.create_carray(f.root, 'indices', tb.Int32Atom(), shape=matrix.indices.shape, filters=filters)
    out[:] = matrix.indices

    #print "saving indptr"
    out = f.create_carray(f.root, 'indptr', tb.Int32Atom(), shape=matrix.indptr.shape, filters=filters)
    out[:] = matrix.indptr

    #print "saving done"

    f.close()

    return

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

    #print "rest sum", rest_sum

    matrix = matrix - floored
    #print matrix.data.shape, floored.data.shape

    #as we can assume that the indices and states are already
    #in random order, we can also assume that ties are handled randomly here.
    #Better randomization might be appropriate though
    idx = matrix.data.argpartition(-rest_sum)[-rest_sum:]

    i, j = matrix.nonzero()

    i_idx = i[idx]
    j_idx = j[idx]

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

def distr_chips_hdf5(file, chips, matrix_sum_final, out_name, norm=True):
    '''
    HDF5 (PyTables) version of the trial roulette method for eliciting Dirichlet priors from
    expressed hypothesis matrix.
    Note that only the informative part is done here.
    This works for densely stored hdf5 matrices (i.e., single data matrix).
    :param file: hdf5 filename where hypothesis matrix A is stored
    :param chips: number of chips C to distribute
    :param matrix_sum_final: the final sum of the input matrix, needs to be pre-calculated
    :param out_name: filename of new file
    :param norm: set False if matrix does not need to be normalized
    :return: True
    '''

    h5 = tb.open_file(file, "r")

    matrix = h5.root.data

    #print matrix[:]

    l = matrix.shape[0]
    k = matrix.shape[1]

    #print matrix[0]

    bl = 1000
    t0= time.time()

    #dtype may need to be altered
    floored = scipy.sparse.lil_matrix((l, k), dtype=np.uint16)
    rest = scipy.sparse.lil_matrix((l, k), dtype=np.float32)
    #print floored.dtype

    matrix_sum = 0.
    nnz_sum = 0.
    flushme = 0
    for i in range(0, l, bl):
        #print i
        if norm:
            rows = matrix[i:min(i+bl, l),:].astype(np.float64) / matrix_sum_final
        else:
            rows = matrix[i:min(i+bl, l),:].astype(np.float64)
        matrix_sum += rows.sum()
        rows = rows * chips
        floor_tmp = np.floor(rows)
        floored[i:min(i+bl, l),:] = floor_tmp
        rest_tmp = rows - floor_tmp
        rest[i:min(i+bl, l),:] = rest_tmp
        #print "nnz floored", floored.nnz
        #print "nnz rest", rest.nnz

        flushme += 1
        #if flushme % 1 == 0:
        #    break
        #     #print "flushing now"
        #     #print (time.time()-t0) / 60.
        #     h5.flush()
        #     #print "flushing done"

        #print (time.time()-t0) / 60.

    #print "looping done"

    floored = floored.tocsr()
    rest = rest.tocsr()

    #print "matrix sum", matrix_sum

    floored_sum = floored.sum()
    #print "floored sum", floored_sum

    rest_sum = int(chips - floored_sum)


    #print "rest sum", rest_sum

    idx = rest.data.argpartition(-rest_sum)[-rest_sum:]

    #print "indexing rest done"

    i, j = rest.nonzero()

    i_idx = i[idx]
    j_idx = j[idx]

    if len(i_idx) > 0:
        floored[i_idx, j_idx] += 1

    del rest

    hdf5_save(floored, out_name)

    h5.close()

    return

def distr_chips_hdf5_sparse(file, chips, matrix_sum_final, out_name, norm=True):
    '''
    HDF5 (PyTables) version of the trial roulette method for eliciting Dirichlet priors from
    expressed hypothesis matrix.
    This version creates a new hdf5 file including the chip distribution.
    Note that only the informative part is done here.
    This works both for sparsely stored hdf5 matrices.
    :param file: hdf5 filename where hypothesis matrix A is stored (needs data, indices, indptr fields)
    :param shape: the shape of the matrix
    :param chips: number of chips C to distribute
    :param matrix_sum_final: the final sum of the input matrix, needs to be pre-calculated
    :param out_name: filename of new file
    :param norm: set False if matrix does not need to be normalized
    :return: True
    '''

    h5 = tb.open_file(file, "r")

    data = h5.root.data
    indices = h5.root.indices
    indptr = h5.root.indptr

    l = data.shape[0]

    bl = 1000
    t0= time.time()

    atom = tb.Atom.from_dtype(data.dtype)

    f = tb.open_file(out_name, 'w')

    filters = tb.Filters(complevel=5, complib='blosc')
    data_out = f.create_carray(f.root, 'data', atom, shape=data.shape, filters=filters)

    #print data.shape
    rest = np.empty(data.shape, dtype=np.float32)

    ##print rest.shape
    #sys.exit()

    #print "saving indices"
    indices_out = f.create_carray(f.root, 'indices', tb.Int32Atom(), shape=indices.shape, filters=filters)
    indices_out[:] = indices[:]

    #print "saving indptr"
    indptr_out = f.create_carray(f.root, 'indptr', tb.Int32Atom(), shape=indptr.shape, filters=filters)
    indptr_out[:] = indptr[:]

    matrix_sum = 0.
    nnz_sum = 0.
    flushme = 0
    floored_sum = 0
    for i in range(0, l, bl):
        #print i
        if norm:
            rows = data[i:min(i+bl, l)].astype(np.float64) / matrix_sum_final
        else:
            rows = data[i:min(i+bl, l)].astype(np.float64)
        matrix_sum += rows.sum()
        rows = rows * chips
        floor_tmp = np.floor(rows)
        data_out[i:min(i+bl, l)] = floor_tmp
        floored_sum += floor_tmp.sum()
        rest_tmp = rows - floor_tmp
        ##print rest_tmp
        ##print rest[i:min(i+bl, l)]
        rest[i:min(i+bl, l)] = rest_tmp
        ##print "nnz floored", floored.nnz
        ##print "nnz rest", rest.nnz

        flushme += 1
        #if flushme % 1 == 0:
        #    break
        #     #print "flushing now"
        #     #print (time.time()-t0) / 60.
        #     h5.flush()
        #     #print "flushing done"

        #print (time.time()-t0) / 60.

    #print "looping done"

    #floored = floored.tocsr()
    #rest = rest.tocsr()

    #print "matrix sum", matrix_sum

    #floored_sum = data_out.sum()
    #print "floored sum", floored_sum

    rest_sum = int(chips - floored_sum)


    #print "rest sum", rest_sum

    idx = rest.argpartition(-rest_sum)[-rest_sum:]

    #print "indexing rest done"

    data_out[idx] += 1

    #print "incrementing index done"

    #floored_sum = data_out.sum()
    #print "final floored sum", floored_sum

    ##print rest.data.shape, data_out.data.shape

    assert(rest.shape == data_out.shape)

    del rest

    h5.close()
    f.close()

    #hdf5_save(floored, "file.h5")

    return