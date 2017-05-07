import falconn as fcn


class LSHIndexer(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def build_index(self, dataset):
        num_data = len(dataset)
        params = fcn.get_default_parameters(num_data, self.dimension)


