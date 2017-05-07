import falconn as fcn
import numpy as np

import timeit


class LSHIndexer(object):
    def __init__(self):
        self.lsh_table = None
        self.data_center = None

    def build_index(self, dataset, preprocess_data=False):
        # const parameters
        num_hash_bits = 20
        num_hash_tables = 30
        num_probes = num_hash_tables

        print('Constructing the LSH table.')
        s_t = timeit.default_timer()

        if preprocess_data:
            dataset = dataset.astype(np.float32)
            self.data_center = np.mean(dataset, axis=0)
            dataset -= self.data_center

        num_data = len(dataset)
        data_dimension = len(dataset[0])
        params = fcn.get_default_parameters(num_data, data_dimension, 'euclidean_squared', False)

        params.l = num_hash_tables
        params.num_rotations = 1
        params.lsh_family = 'cross_polytope'

        fcn.compute_number_of_hash_functions(num_hash_bits, params)

        self.lsh_table = fcn.LSHIndex(params)
        self.lsh_table.setup(dataset)
        self.lsh_table.set_num_probes(num_probes)

        s_e = timeit.default_timer()
        print('Done')
        print('Construction time: {}'.format(s_e - s_t))

    def find_k_nearest_neighbors(self, query, k=1):
        if self.data_center is not None:
            query -= self.data_center

        if k == 1:
            return [self.lsh_table.find_nearest_neighbor(query)]
        else:
            return self.lsh_table.find_k_nearest_neighbors(query, k)

# Test
# dim = 256
# indexer = LSHIndexer(dim)
# data = np.zeros([10, dim], np.float32)
# indexer.build_index(data)



