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

def build_flat_l2(build_data , **fixed_params):
    dim = build_data.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.train(build_data)
    index.add(build_data)
    return index


def build_faiss_cosine(build_data, **fixed_params):
    dim = build_data.shape[1]
    faiss.normalize_L2(build_data.astype(np.float32))
    index = faiss.IndexFlatIP(dim)
    index.train(build_data)
    index.add(build_data)
    return index


def search_flat(index, query_data, k):
    distances, labels = index.search(x=query_data, k=k)
    return distances, labels

def search_faiss_cosine(index, query_data, k, nprobe=1):
    faiss.normalize_L2(query_data.astype(np.float32))
    index.nprobe = nprobe
    distances, labels = index.search(query_data, k)
    return distances, labels

def search_faiss(index, query_data, k, nprobe=1):
    index.nprobe = nprobe
    distances, labels = index.search(query_data, k)
    return distances, labels
