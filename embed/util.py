import numpy as np


def partition_num(total, n):
	if total % n == 0:
		return [total//n]*n
	return [total//n]*n + [total%n]

def create_alias_table(probs):
	N = len(probs)
	accept, alias = [0] * N, [0] * N
	small, large = [], []

	for ix, prob in enumerate(probs):
		accept[ix] = N * prob
		if accept[ix] < 1:
			small.append(ix)
		else:
			large.append(ix)

	while small and large:
		small_idx = small.pop()
		large_idx = large.pop()

		alias[small_idx] = large_idx
		accept[large_idx] = accept[large_idx] + accept[small_idx] - 1
		if accept[large_idx] < 1:
			small.append(large_idx)
		else:
			large.append(large_idx)

	while large:
		large_idx = large.pop()
		accept[large_idx] = 1
	while small:
		small_idx = small.pop()
		accept[small_idx] = 1
	return accept, alias

def alias_sample(accept, alias):
	N = len(accept)
	ix = int(np.random.random()*N)
	if np.random.random() < accept[ix]:
		return ix
	return alias[ix]