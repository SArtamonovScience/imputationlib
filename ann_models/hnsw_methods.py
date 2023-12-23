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
