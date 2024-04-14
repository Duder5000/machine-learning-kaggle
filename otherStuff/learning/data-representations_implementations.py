import numpy as np

"""
Add, subtract, multiply, divide, relu
Broadcasting (adding matrice & vectors)
Vector-vector dot, matrix-vector dot, matrix-matrix dot
"""

def add_tensors(x, y):
    sum_tensor = y.copy()

    for i in range(x.shape[0]):
        for n in range(x.shape[1]):
            sum_tensor[i][n] += x[i][n]
    return sum_tensor

def subtract_tensors(x, y):
    difference_tensor = y.copy()

    for i in range(x.shape[0]):
        for n in range(x.shape[1]):
            difference_tensor[i][n] -= x[i][n]
    return difference_tensor

def multiply_tensors(x, y):
    product_tensor = y.copy()

    for i in range(x.shape[0]):
        for n in range(x.shape[1]):
            product_tensor[i][n] *= x[i][n]
    return product_tensor

def divide_tensors(x, y):
    quotient_tensor = y.copy()

    for i in range(x.shape[0]):
        for n in range(x.shape[1]):
            quotient_tensor[i][n] /= x[i][n]
    return quotient_tensor

def relu_tensors(x):
    relu_tensor = x.copy()
    for i in range(relu_tensor.shape[0]):
        for n in range(relu_tensor.shape[1]):
            relu_tensor[i][n] = max(relu_tensor[i][n], 0)
    return relu_tensor

def vector_to_vector_dot(x, y):
    final_sum = 0

    for i in range(len(x)):
        final_sum += x[i] * y[i]
    return final_sum

tensor1 = np.array([[1, 2, 3],
                    [1, 2, 3]])

tensor2 = np.array([[4, 5, 6],
                    [4, 5, 6]])

matrice = np.array([[5, 78, 2, 34, 0],
                    [6, 79, 3, 35, 1],
                    [7, 80, 4, 36, 2]])

matrice2 = np.array([[5, 78, 2, 34, 0],
                    [6, 79, 3, 35, 1],
                    [7, 80, 4, 36, 2]])

vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([1, 2, 3, 4])

print(add_tensors(tensor1, tensor2))
#print(vector_to_vector_dot(vector1, vector2))

# Tensor reshaping

print(tensor1.shape) # (2, 3)

tensor1 = tensor1.reshape(6, 1)

print(tensor1.shape) # (6, 1)








