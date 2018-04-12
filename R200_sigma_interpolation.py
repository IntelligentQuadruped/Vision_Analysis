import scipy.interpolate as interpolate
import matplotlib.pyplot as plt 
import numpy as np
import seaborn


x_data = [0.69870297302827089,0.95327826619912592,1.7319821856332107,2.482256090892522]
y_data = [0.0066268269500679563,0.0030826634806626831,0.014238865442736092,0.035892770702646355]

y1 = interpolate.interp1d(x_data,y_data,kind='quadratic',fill_value='extrapolate')
x_new = np.arange(0,4,0.1)
weights = np.polyfit(x_new,y1(x_new),6)
a,b,c,d,e,f,g = weights
y2 = lambda x : a*x**6 + b*x**5 +c*x**4 +d*x**3 + e*x**2 + f*x + g
print(weights)
plt.figure("std_vs_measured_d")
plt.plot(x_data,y_data,'o',x_new,y1(x_new),'--')#,x_new,y2(x_new))
plt.xlabel('Measured Distance [meter]')
plt.ylabel('Standard Deviation at given Distance')
plt.show()