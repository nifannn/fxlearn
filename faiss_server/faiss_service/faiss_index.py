import faiss
import numpy as np


class FaissIndex(object):
    """

    """
    def __init__(self, id2vec):
        self._set_id2vec(id2vec)

    def _set_id2vec(self, id2vec):
        self._ix2id = list(id2vec.keys())
        self._id2ix = {item_id:ix for ix, item_id in enumerate(self._ix2id)}
        self._id2vec = id2vec
        matrix = np.zeros((len(self._ix2id), len(self._id2vec[self._ix2id[0]])))
        for ix, item_id in enumerate(self._ix2id):
            matrix[ix] = self._id2vec[item_id]
        matrix = matrix.astype('float32')
        self._index = faiss.IndexFlatL2(len(self._id2vec[self._ix2id[0]]))
        self._index.add(matrix)

    def _search(self, vectors, k):
        if vectors.size > 0:
            return self._index.search(vectors, k)
        return ([], [])

    def search_by_ids(self, ids, k):
        vectors = [self._id2vec[item_id] for item_id in ids]
        vectors = np.array(vectors, dtype='float32')
        vectors = np.atleast_2d(vectors)
        distances, result_idxes = self._search(vectors, k+1)
        results = {item_id:[{self._ix2id[ix]: float(distances[row, col])} 
                   for col, ix in enumerate(result_idxes[row].tolist()) 
                   if ix != self._id2ix[item_id] and ix >= 0] 
                   for row, item_id in enumerate(ids)}
        return results

    def search_by_vectors(self, vectors, k):
        vectors = np.array(vectors, dtype='float32')
        vectors = np.atleast_2d(vectors)
        distances, result_idxes = self._search(vectors, k)
        results = {row:[{self._ix2id[ix]: float(distances[row, col])} 
                   for col, ix in enumerate(result_idxes[row].tolist())
                   if ix >= 0] 
                   for row in range(len(result_idxes))}
        return results