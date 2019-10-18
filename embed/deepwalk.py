import networkx as nx
import joblib
import random
from . import word2vec
from . import util


class DeepWalk(object):
	"""
	DeepWalk
	-----------------------
	Refer to the paper:
	DeepWalk: Online Learning of Social Representations
	Bryan Perozzi, Rami Al-Rfou and Steven Skiena
	Knowledge Discovery and Data Mining (KDD), 2014
	"""
	def __init__(self, num_walks=10, walk_length=80, embed_dim=128, 
					   window_size=10, min_count=3, min_len=3, max_len=200,
					   random_window=False, n_sampled=100, subsample=True, 
					   subsample_thr=1e-5, lr=None, name=None, log_file=None):
		self._walk_length = walk_length
		self._num_walks = num_walks
		self._embed_dim = embed_dim
		self._window_size = window_size
		self._min_count = min_count
		self._min_len = min_len
		self._max_len = max_len
		self._random_window = random_window
		self._n_sampled = n_sampled
		self._subsample = subsample
		self._subsample_thr = subsample_thr
		self._lr = lr
		self._graph = None
		self._word2vec = None
		self._log_file = log_file
		self._set_name(name)

	def _read_edgelist(self, fpath):
		self._graph = nx.read_edgelist(fpath, create_using=nx.DiGraph())

	def _random_walk(self, walk_length, start_node):
		walk = [start_node]
		while len(walk) < walk_length:
			cur_node = walk[-1]
			nbrs = list(self._graph.neighbors(cur_node))
			if len(nbrs) > 0:
				walk.append(random.choice(nbrs))
			else:
				break
		return walk

	def _set_name(self, name):
		self.name = "DeepWalk" if name is None else name

	def _simulate_walks(self, num_walks, walk_length):
		walks = []
		nodes = list(self._graph.nodes())
		for _ in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self._random_walk(self._walk_length, node))
		return walks

	def _generate_sequences(self, n_jobs=4):
		sequences = joblib.Parallel(n_jobs=n_jobs)(
					joblib.delayed(self._simulate_walks)(num, self._walk_length)
					for num in util.partition_num(self._num_walks, n_jobs))
		return sequences

	def get_embeddings(self):
		if self._word2vec is None:
			return dict()
		return self._word2vec.get_embeddings()

	def train(self, edgelist_path, epcohs=10, batch_size=128, n_jobs=4):
		self._read_edgelist(edgelist_path)
		sequences = self._generate_sequences(n_jobs)
		self._word2vec = word2vec.SGNS(self._embed_dim, self._window_size, self._min_count,
									   self._min_len, self._max_len, self._random_window, self._n_sampled,
									   self._subsample, self._subsample_thr, self._lr, self.name, self._log_file)
		self._word2vec.train(sequences, epcohs=epcohs, batch_size=batch_size)
		return self

	def save(self, out_path, field_separator="\t", vector_separator=" "):
		self._word2vec.save(out_path, field_separator, vector_separator)