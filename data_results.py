"""
Script to gather data 
"""

import numpy as np 
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn

data_dir = ["./data","./sample_obstacle_course"]
# data_dir = ['./windmachine']

data = []
for data_path in data_dir:
    for f in os.listdir(data_path):
        if "d" in f:
            try:
                path = os.path.join(data_path,f)
                matrix = np.load(path)
                matrix[matrix > 4000] = 0.0
                nan = len(matrix[matrix < 1])
                total = len(matrix.flatten())
                result = 1 - nan/total
                data.append(result)
                # if True:
                #     plt.figure()
                #     plt.title(f)
                #     plt.imshow(matrix)
                #     plt.show()
            except TypeError:
                path = os.path.join(data_path,f)
                d= np.load(path)
                # for i in range(5):
                #     s = 'arr_{}'.format(i+1)
                s = 'arr_1'
                matrix = d[s]
                nan = len(matrix[matrix < 1])
                total = len(matrix.flatten())
                result = 1 - nan/total
                data.append(result)
                d.close()
data.sort()
data  = np.array(data,dtype = float)
# delete outliers that are outside 3 sigma
data = data[abs(data - np.mean(data)) < 3 * np.std(data)]
# normalize data 
norm = len(data)
# normed_data = np.divide(data,norm)
mu = np.mean(data)
sigma = np.std(data)


weights = np.ones_like(data)/float(len(data))

plt.figure("depth_density")
n,bins,patches = plt.hist(data,bins=100,weights=weights,alpha=0.75,normed=True)
plt.xlabel('Data Density')
plt.ylabel('Probability of Data Density')
plt.title('Depth Sensor Performance: n = %d, mu = %.2f, sigma = %.3f'%(len(data),mu,sigma))
plt.show()

plt.figure("depth_density_normal")
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.xlabel('Data Density')
plt.ylabel('Probability of given Data Density')
plt.title('Normal Distribution: mu = %.2f, sigma = %.3f'%(mu,sigma))
plt.show()
print("Average depth data collected: %2.4f"%(mu))
print("With a standard deviation: %2.4f"%(sigma))







