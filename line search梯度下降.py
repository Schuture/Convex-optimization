import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

'''
np.random.seed(666)
a = np.random.rand(10,10)
b = np.random.normal(loc = 10, scale = 1, size = (10, 1))
c = np.random.rand(10).reshape((10,1))
'''
a = np.ones((10,10)) * -1
b = np.ones((10,1)) * 5
c = np.ones((10,1)) * 1


def f(x):
    S = 0
    for i in range(10):
        S += np.log(b[i] - a[:, i].dot(x))
    return c.T.dot(x) - S


def grad(x):
    '''
    gradient at point x
    '''
    grad = c.copy()
    for i in range(10):
        grad += a[:, i].reshape((10,1)) / (b[i] - a[:,i].dot(x))
    return grad


def F(alpha):
    '''
    F(alpha) = f(x - alpha * grad(x)), for exact line search
    '''
    S = 0
    for i in range(10):
        S += np.log(b[i] - ax[i] + alpha * agrad[i])
    return -alpha * float(cgrad) + float(cx) - S
    

def gradientDecsent(f, grad, F, alpha, beta, Iterations, back_track = True):
    '''
    Gradient decsent for function f
    
    Input:
        f: function to optimize
        grad: gradient function for f
        F: F(alpha) = f(x + alpha * grad(x)), for exact line search
        alpha: initial learning rate for line search
        beta: Attenuation coefficient
    Output:
        opt_value: optimal value
        gra: a list, the gradient of every step
        value: alist, the value of every step
        k: the steps of iteration
    '''
    x = np.zeros((10,1))
    gra = np.zeros(Iterations)
    value = np.zeros(Iterations)
    
    for k in range(1, Iterations):
        grad1 = grad(x) 
        gra[k] = np.linalg.norm(grad1) # 梯度的范数
        value[k] = f(x)
        
        if back_track:
            t = 2
            x_plus = x - alpha * t * grad1
            while f(x) - f(x_plus) < -alpha * t * grad1.T.dot(grad1): # backtracking
                t = beta * t
                x_plus = x - alpha * t * grad1
            x = x_plus
        else: # exact
            global cgrad
            global cx
            global ax
            global agrad
            cgrad = c.T.dot(grad1)
            cx = c.T.dot(x)
            ax = []
            agrad = []
            for i in range(10):
                ax.append(a[:, i].T.dot(x))
                agrad.append(a[:, i].T.dot(grad1))
            alpha = minimize_scalar(F, method = 'brent')
            x = x - alpha.x * grad1
    
        if k % 10 == 0:
            print('Iteration: {}. The norm of gradient now is: {:.7f}'.format(k, gra[k]))
            
        if np.linalg.norm(grad1) <= 1e-7:
            print('Gradient has converged!')
            break
        
    return min(value), gra, value, k


if __name__ == '__main__':
    # 设置
    back_track = True
    Iterations = 500
    alpha = 0.1
    beta = 0.5
    optimal_value, gra, value, k = gradientDecsent(f, grad, F, alpha, beta, Iterations, back_track)

    # 梯度变化图
    plt.figure(figsize = (15, 10))
    plt.semilogy(list(range(1, k+1)), gra[1:k+1])
    plt.title('norm of gradient of gradient decent', fontsize = 18)
    plt.xlabel('iteration', fontsize = 18)
    plt.ylabel('norm of gradient', fontsize = 18)
    plt.show()
    
    # 函数值下降图
    plt.figure(figsize = (15, 10))
    plt.semilogy(list(range(1, k+1)), value[1:k+1] - optimal_value + 1e-6)
    plt.title('function value (f(x) - p*) of gradient decent', fontsize = 18)
    plt.xlabel('iteration', fontsize = 18)
    plt.ylabel('(f(x) - p*)', fontsize = 18)
    plt.show()