def partition_num(total, n):
	if total % n == 0:
		return [total//n]*n
	return [total//n]*n + [total%n]