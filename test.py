from neural.optimized import networks


def main():
	net = networks.Network((2, 3, 1))
 	net.compile()

	inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
	targets = [(0,), (1,), (1,), (0,)]

	net.train(inputs, targets, 1000, alpha=0.1)

	for inp in inputs:
		print(f'{inp} -> {net.forward(inp)}')


if __name__ == '__main__':
	main()
