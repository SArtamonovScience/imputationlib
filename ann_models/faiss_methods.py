import faiss

def build_IVFPQ(build_data, coarse_index, nlist, m, nbits, metric, num_threads=1):
    dim = build_data.shape[1]
    faiss.omp_set_num_threads(num_threads)
    
    index = faiss.IndexIVFPQ(
        coarse_index,
        dim,
        nlist,
        m,
        nbits,
        metric
    )
    index.train(build_data)
    index.add(build_data)
    return index

def build_IVFFlat(build_data, coarse_index, nbits, metric, num_threads=1):\
    dim = build_data.shape[1]
    faiss.omp_set_num_threads(num_threads)
    
    index = faiss.IndexIVFFlat(
        coarse_index,
        dim,
        nlist,
        metric
    )
    index.train(build_data)
    index.add(build_data)
    return index
