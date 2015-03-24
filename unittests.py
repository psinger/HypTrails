from __future__ import division

__author__ = 'psinger'

import random
import unittest
from scipy.sparse import rand, lil_matrix, csr_matrix
from sklearn.preprocessing import normalize
from src.trial_roulette import *
import os

class TestRouletteFunctions(unittest.TestCase):

    def setUp(self):
        self.states = 100
        self.matrix = rand(self.states,self.states, format='csr')

    def test_distr_chips(self):
        ret = distr_chips(self.matrix, self.states*self.states)

        self.assertEqual(ret.sum(), self.states*self.states)

    def test_distr_chips_row(self):
        tmp = lil_matrix(self.matrix.shape)
        for i in xrange(self.states):
            tmp[i,:] = distr_chips(self.matrix[i,:], self.states)

        self.assertEqual(tmp.sum(), self.states*self.states)

    def test_distr_chips_row_strange_chips(self):
        tmp = lil_matrix(self.matrix.shape)
        for i in xrange(self.states):
            tmp[i,:] = distr_chips(self.matrix[i,:], self.states+53)

        self.assertEqual(tmp.sum(), (self.states)*(self.states)+ self.states*53)

    def test_distr_chips_hdf5(self):
        filters = tb.Filters(complevel=5, complib='blosc')
        atom = tb.Atom.from_dtype(self.matrix.dtype)
        f = tb.open_file("test.hdf5", 'w')
        out = f.create_carray(f.root, 'data', atom, shape=self.matrix.shape, filters=filters)
        out[:] = self.matrix.toarray()
        f.close()

        ret1 = distr_chips(self.matrix, self.states*self.states)
        distr_chips_hdf5("test.hdf5", self.states*self.states, self.matrix.sum(), "out.hdf5")

        h5 = tb.open_file("out.hdf5", 'r')

        ret2 = csr_matrix((h5.root.data[:], h5.root.indices[:], h5.root.indptr[:]), shape=self.matrix.shape, dtype=np.float64)

        np.testing.assert_array_equal(ret1.toarray(), ret2.toarray())

        h5.close()
        os.remove("test.hdf5")
        os.remove("out.hdf5")

    def test_distr_chips_hdf5_sparse(self):
        hdf5_save(self.matrix,"test2.hdf5")

        ret1 = distr_chips(self.matrix, self.states*self.states)
        distr_chips_hdf5_sparse("test2.hdf5", self.states*self.states, self.matrix.sum(), "out2.hdf5")

        h5 = tb.open_file("out2.hdf5", 'r')

        ret2 = csr_matrix((h5.root.data[:], h5.root.indices[:], h5.root.indptr[:]), shape=self.matrix.shape, dtype=np.float64)

        np.testing.assert_array_equal(ret1.toarray(), ret2.toarray())

        h5.close()
        os.remove("test2.hdf5")
        os.remove("out2.hdf5")

if __name__ == '__main__':
    unittest.main()