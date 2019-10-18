import networkx as nx
from . import word2vec
from . import util


class Node2Vec(object):
	"""
	Node2Vec
	-------------------------------
	Refer to the paper:
	node2vec: Scalable Feature Learning for Networks
	Aditya Grover and Jure Leskovec 
	Knowledge Discovery and Data Mining (KDD), 2016
	"""
	def __init__(self, p=1, q=2, num_walks=10, walk_length=80,
				 embed_dim=128, window_size=10, min_count=3,
				 min_len=3, max_len=200, random_window=False,
				 n_sampled=100, subsample=True, subsample_thr=1e-5,
				 lr=None, name=None, log_file=None):
		self._p = p
		self._q = q
		self._num_walks = num_walks
		self._walk_length = walk_length
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
		self.name = name if name is not None else "Node2Vec"
		self._log_file = log_file
		self._graph = None
		self._word2vec = None

	def _get_alias_edges(self):
		pass

	def _get_alias_nodes(self):
		pass

	def _preprocess_transition_probs(self):
		pass

	def _node2vec_walk(self):
		pass

	def _read_edgelist(self, fpath):
		self._graph = nx.read_edgelist(fpath, create_using=nx.DiGraph())

	def _simulate_walks(self):
		pass

	def get_embeddings(self):
		pass

	def train(self):
		pass

	def save(self):
		pass
		