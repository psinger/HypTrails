from __future__ import division

__author__ = 'psinger'

import tables as tb
import time
import scipy
import numpy as np
import sys

def distr_chips(matrix, chips):
    '''
    Trial roulette method for eliciting Dirichlet priors from
    expressed hypothesis matrix.
    Note that only the informative part is done here.
    :param matrix: csr_matrix A_k expressing theory H_k
    :param chips: number of overall (whole matrix) chips C to distribute
    :return: Dirichlet pseudo clicks in the shape of a matrix
    '''

    print "chips", chips

    chips = chips

    if float(chips).is_integer() == False:
        raise Exception, "Only use C = |S|^2 * k"

    #it may make sense to do this in the outer scripts for memory reasons
    matrix = (matrix / matrix.sum()) * chips

    print "matrix nnz", matrix.nnz
    print matrix.max()


    print "normalization done"
    print "matrix nnz", matrix.nnz
    print matrix.max()

    floored = matrix.floor()

    print "flooring done"

    print "floored sum", floored.sum()


    rest_sum = int(chips - floored.sum())

    print "rest sum", rest_sum

    matrix = matrix - floored
    print matrix.data.shape, floored.data.shape


    idx = matrix.data.argpartition(-rest_sum)[-rest_sum:]

    i, j = matrix.nonzero()

    i_idx = i[idx]
    j_idx = j[idx]

    if len(i_idx) > 0:
        floored[i_idx, j_idx] += 1

    print "final sum", floored.sum()

    print matrix.data.shape, floored.data.shape

    #assert(matrix.data.shape == floored.data.shape)

    floored.eliminate_zeros()

    print type(floored)

    del matrix

    print "prior calc done"

    print floored.nnz

    return floored

def distr_chips_row(matrix, chips):
    '''
    Trial roulette method for eliciting Dirichlet priors from
    expressed hypothesis matrix.
    This method works for single row matrices with the condition that
    each row is the same.
    Note that only the informative part is done here.
    :param matrix: csr_matrix A_k expressing theory H_k
    :param chips: number of overall (whole matrix) chips C to distribute
    :return: Dirichlet pseudo clicks in the shape of a matrix
    '''



    length = matrix.shape[1]

    chips = chips / length

    print "chips", chips

    if float(chips).is_integer() == False:
        raise Exception, "Only use C = |S|^2 * k"

    sum = matrix.sum()
    #sum *= len


    #it may make sense to do this in the outer scripts for memory reasons
    matrix = (matrix / sum) * chips

    print matrix

    print "matrix nnz", matrix.nnz
    print matrix.max()


    print "normalization done"
    print "matrix nnz", matrix.nnz
    print matrix.max()

    floored = matrix.floor()

    print "flooring done"

    print "floored sum", floored.sum()


    rest_sum = int(chips - floored.sum())

    print "rest sum", rest_sum

    matrix = matrix - floored
    print matrix.data.shape, floored.data.shape

    print matrix

    idx = matrix.data.argpartition(-rest_sum)[-rest_sum:]

    i, j = matrix.nonzero()

    print i, j, idx
    #sys.exit()

    i_idx = i[idx]
    j_idx = j[idx]

    #print type(i_idx), j_idx, len(j_idx)

    if len(i_idx) > 0:
        floored[i_idx, j_idx] += 1

    print "final sum", floored.sum()

    print matrix.data.shape, floored.data.shape

    #assert(matrix.data.shape == floored.data.shape)

    floored.eliminate_zeros()

    print type(floored)

    del matrix

    print "prior calc done"

    print floored.nnz

    return floored

def hdf5_save(matrix, filename, dtype=np.dtype(np.float64)):
    '''
    Helper function for storing scipy matrices as PyTables HDF5 matrices
    see http://www.philippsinger.info/?p=464 for further information
    :param matrix: matrix to store
    :param filename: filename for storage
    :param dtype: dtype
    :return: True
    '''

    print matrix.shape

    atom = tb.Atom.from_dtype(dtype)

    f = tb.open_file(filename, 'w')

    print "saving data"
    filters = tb.Filters(complevel=5, complib='blosc')
    out = f.create_carray(f.root, 'data', atom, shape=matrix.data.shape, filters=filters)
    out[:] = matrix.data

    print "saving indices"
    out = f.create_carray(f.root, 'indices', tb.Int32Atom(), shape=matrix.indices.shape, filters=filters)
    out[:] = matrix.indices

    print "saving indptr"
    out = f.create_carray(f.root, 'indptr', tb.Int32Atom(), shape=matrix.indptr.shape, filters=filters)
    out[:] = matrix.indptr

    print "saving done"

    f.close()

    return

def distr_chips_hdf5(file, chips, matrix_sum_final):
    '''
    HDF5 (PyTables) version of the
    trial roulette method for eliciting Dirichlet priors from
    expressed hypothesis matrix.
    Note that only the informative part is done here.
    :param file: hdf5 filename where hypothesis matrix A is stored
    :param chips: number of chips C to distribute
    :param matrix_sum_final: the final sum of the input matrix, needs to be pre-calculated
    :return: True
    '''

    h5 = tb.open_file(file, "r")

    matrix = h5.root.data

    l = matrix.shape[0]

    bl = 1000
    t0= time.time()

    #dtype may need to be altered
    floored = scipy.sparse.lil_matrix((l, l), dtype=np.uint16)
    rest = scipy.sparse.lil_matrix((l, l), dtype=np.float32)
    print floored.dtype

    matrix_sum = 0.
    nnz_sum = 0.
    flushme = 0
    for i in range(0, l, bl):
        print i
        rows = matrix[i:min(i+bl, l),:].astype(np.float64) / matrix_sum_final
        matrix_sum += rows.sum()
        rows = rows * chips
        floor_tmp = np.floor(rows)
        floored[i:min(i+bl, l),:] = floor_tmp
        rest_tmp = rows - floor_tmp
        rest[i:min(i+bl, l),:] = rest_tmp
        print "nnz floored", floored.nnz
        print "nnz rest", rest.nnz

        flushme += 1
        #if flushme % 1 == 0:
        #    break
        #     print "flushing now"
        #     print (time.time()-t0) / 60.
        #     h5.flush()
        #     print "flushing done"

        print (time.time()-t0) / 60.

    print "looping done"

    floored = floored.tocsr()
    rest = rest.tocsr()

    print "matrix sum", matrix_sum

    floored_sum = floored.sum()
    print "floored sum", floored_sum

    rest_sum = int(chips - floored_sum)


    print "rest sum", rest_sum

    idx = rest.data.argpartition(-rest_sum)[-rest_sum:]

    print "indexing rest done"

    floored.data[idx] += 1

    print "incrementing index done"

    floored_sum = floored.sum()
    print "final floored sum", floored_sum

    print rest.data.shape, floored.data.shape

    assert(rest.data.shape == floored.data.shape)

    del rest

    hdf5_save(floored, "file.h5")

    return
