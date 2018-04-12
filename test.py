from depth_completion import depth_completion
import numpy as np
import matplotlib.pyplot as plt
import time 
import os

data_dir = "./sample_data_5"

for f in os.listdir(data_dir):
	if "c" in f:
		try:
			path = os.path.join(data_dir,f)
			matrix = np.load(path)
			plt.imshow(matrix)
			plt.title(f)
			plt.show()
		except KeyboardInterrupt:
			break