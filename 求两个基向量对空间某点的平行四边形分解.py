'''
For getting the projection of c on the hyperplane formed by a and b,
or the best linear combination of vectors a and b to reach c, 
we can directly minimize norm(ax+by-c)^2, the norm is 2-norm for simplicity
After derivation, the problem can be seen as solving a linear equation
'''

import numpy as np

# ax + by has the minimal distance to c
a = np.array([1,1.32,0.456,0.3,2.5])
b = np.array([0.345,1,2.45,5,123.4])
c = np.array([4.4,7.5,1.123,9.87,4])
print('We want to find the best approximation of {} \nby {} and {}\n'.format(c,a,b))

matrix = np.matrix([[a.dot(a),a.dot(b)],[a.dot(b),b.dot(b)]])
vector = np.array([a.dot(c), b.dot(c)]).reshape((2,-1))

solution = matrix.I.dot(vector)
x = float(solution[0])
y = float(solution[1])
combination = x * a + y * b

print('The best approximation for c = {} is: \
    \n{} * a + {} * b = {}'.format(c, x, y, combination))

edge1 = np.linalg.norm(combination)
edge2 = np.linalg.norm(c - combination)
edge3 = np.linalg.norm(c)

print('\nWe want to see if c-ax-by is perpendicular to ax+by.')
print('Let c-ax-by and ax+by be the two short edges of a triangle, then:')
print('edge1^2 + edge2^2 = {};'.format(round(edge1**2 + edge2**2,4)),end='   ')
print('c^2 = {}'.format(round(edge3**2, 4)))
print('They are the same! We can say we have found the projection point ax+by!')

print('In additional, c-ax-by dot a = {}'.format((c-combination).dot(a)))
print('In additional, c-ax-by dot b = {}'.format((c-combination).dot(b)))
print('This means c-ax-by is perpendicular to the hyperplane formed by a, b')