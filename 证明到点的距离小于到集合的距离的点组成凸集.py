import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 1000)
sin_theta = np.sin(theta)
cos_theta = np.cos(theta)
r = 2 * (1 - sin_theta)
x = r * cos_theta
y = r * sin_theta

# the original heart shape
print('This is the original shape of the set:')
plt.figure(figsize = (12, 13.5))
plt.axis([-3,5,-5,4])
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print('This is the graph to show the aiming region:')
plt.figure(figsize = (12, 13.5))
plt.axis([-3,5,-5,4])
plt.scatter(x,y)

# point
point = np.array([1,3]) # the point outside the set
plt.scatter(point[0], point[1], color = 'red')

# lines, y = kx + b
for i in range(1000):
    x_ = np.linspace(-3,5,2)
    k = (x[i] - point[0]) / (point[1] - y[i])
    b = (y[i] + point[1])/2 - k * (x[i] + point[0])/2
    y_ = k * x_ + b
    plt.plot(x_, y_)

plt.xlabel('x')
plt.ylabel('y')
plt.show()

print('The region is the blank part around the red point.\n We notice that it is a convex set.')












