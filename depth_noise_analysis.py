import numpy as np 
import os
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def cutter(matrix):
    matrix = matrix[200:400,200:400]
    return matrix

COLOR = False
DEPTH = False
HISTOGRAM = False
NORMAL = False

hor = 215.9
ver = 355.6
offset = 20
data_dir = ["./data/sample_data_2","./data/sample_data_1","./data/sample_data_4","./data/sample_data_5"]
distance = [2*ver,2*ver+hor,4*ver+hor,6*ver+hor]

data = []
for i,data_path in enumerate(data_dir):
    for f in os.listdir(data_path):
        if "d" in f:
            path = os.path.join(data_path,f)
            matrix = np.load(path)
            matrix[matrix > 4000] = 0.0
            matrix = cutter(matrix)
            if COLOR:
                mpl.style.use('default')
                f_c = f.replace("d","c")
                path_c = os.path.join(data_path,f_c)
                color = np.load(path_c)
                color = cutter(color)
                plt.figure("depth_sample_{}_color".format(i+1))
                plt.imshow(color)
                plt.title("RGB")
                plt.xlabel('x-axis [pixel]')
                plt.ylabel('y-axis [pixel]')
                plt.grid(False)
                plt.show()
            if DEPTH:
                plt.figure("depth_sample_{}_depth".format(i+1))
                plt.imshow(matrix)
                plt.title("Depth")
                plt.xlabel('x-axis [pixel]')
                plt.ylabel('y-axis [pixel]')
                plt.grid(False)
                plt.show()

            matrix = matrix.flatten()
            matrix = matrix[matrix > 0.0]
            matrix = matrix/1000
            mu = np.mean(matrix)
            sigma = np.std(matrix)
            data.append([mu,sigma])
            if HISTOGRAM:
                mpl.style.use('seaborn')
                print('Mean: {}, Std: {}'.format(mu, sigma))
                matrix = matrix[abs(matrix - mu) < 3 * sigma]
                plt.figure("depth_sample_{}_histogram".format(i+1))
                n,bins,patches = plt.hist(matrix,bins=50,alpha=0.75)
                plt.xlabel('Depth Value [meter]')
                plt.ylabel('Occurences of Depth value')
                plt.title('Reference Distance: %.2f meter'%(round(distance[i]/1000,2)))
                plt.show()
            if NORMAL:
                mpl.style.use('seaborn')
                plt.figure("depth_sample_{}_normal".format(i+1))
                plt.xlabel('Depth Value [meter]')
                plt.ylabel('Probability Depth Value')
                plt.title('Normal Distribution: mu = %.3f meter, sigma = %.4f meter'%(mu,sigma))
                y = mlab.normpdf( bins, mu, sigma)
                l = plt.plot(bins, y, 'r--', linewidth=1)
                plt.show()
            break

mpl.style.use('seaborn')
measured = [x[0] for x in data]
distance = [round(x/1000,2) for x in distance]
standard_error = [abs(x[0]-distance[i]) for i,x in enumerate(data)]
plt.figure("depth_accuracy")
plt.title('Depth Accuracy')
plt.xlabel('Reference Depth [meter]')
plt.ylabel('Sensor Depth [meter]')
plt.errorbar(distance,measured,yerr=standard_error,fmt = '--.',ecolor='r', capsize=4,capthick=2)
plt.show()


print(data)
print(distance)
print(standard_error)
