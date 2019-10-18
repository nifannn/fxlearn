import networkx as nx


class GraphEmbedding(object):
	"""GraphEmbedding"""
	def __init__(self):
		pass

	def _read_edgelist(self, fpath, weighted=False, directed=True):
		if weighted:
			G = nx.read_edgelist(fpath, data=(('weight',float),), create_using=nx.DiGraph())
		else:
			G = nx.read_edgelist(fpath, create_using=nx.DiGraph())

		if not directed:
			G = G.to_undirected()

		self._graph = G
		


class Word2VecBasedGraphEmbedding(GraphEmbedding):
	"""Word2Vec Based Graph Embedding"""
	def __init__(self):
		pass