import hnswlib

def build_hnsw(build_data, space='l2', M=32, ef_construction=32):
    """
      Builds hnsw ANN Index.
      params:
        space: 'l2' or 'cosine'
        ...
    """
    
    space = space
    M = M
    dim = build_data.shape[-1]
    ef_construction = ef
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=build_data.shape[0], 
                     ef_construction=ef_construction, 
                     M=M)
    index.add_items(np.float32(build_data), 
                    np.arange(build_data.shape[0]))
    index.train(build_data)
    return index

def search_hnsw(index, query_data, k, efSearch=10):
    """
    Searches k approximate neighbors in hnsw index.
    """
    index.set_ef(efSearch)
    labels, distances = index.knn_query(np.float32(query_data), k=k)
    return distances, labels

class HNSWSearcher(object):
    def __init__(self, space='l2', M=32, ef_construction=32):
        self.index = None
        self.dim = None
        self.space = space
        self.M = M
        self.ef_construction = ef_construction

    def fit(self, X):
        self.index = build_hnsw(X, space=self.space, M=self.M, ef_construction=self.ef_construction)
        return self

    def kneighbors(self, X, k, efSearch=10):
        if self.index is None:
            raise ValueError("Unfitted")
        distances, labels = search_hnsw(self.index, X, k, efSearch)
