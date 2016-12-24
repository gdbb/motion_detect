import numpy as np


def product_weight():
	size = 16
	circle_num = size / 2
	circle_block_num = -4

	weight = np.empty((size, size))
	num = 0

	for index in reversed(range(circle_num)):
		m = index
		n = size - 1 - index

		circle_block_num += 8
		value = 1. / circle_num / circle_block_num

		for i in range(size):
			weight[m][i] = value
			weight[n][i] = value
			weight[i][m] = value
			weight[i][n] = value

	print weight

def product_num():
	size = 16
	for i in range(size):
		for j in range(size):
			print i, j

if __name__ == "__main__":
	#product_num()
	product_weight()