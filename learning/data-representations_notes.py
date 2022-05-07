import numpy as np

"""
Data Representations for Neural Networks
"""
# Scalars (OD tensors)
scalar = np.array(4)

# Vectors (1D tensors)
vector = np.array([1, 3, 6, 19, 4])

# Matrices (2D tensors)
matrice = np.array([[5, 78, 2, 34, 0],
                    [6, 79, 3, 35, 1],
                    [7, 80, 4, 36, 2]])

matrice2 = np.array([[5, 78, 2, 34, 0],
                    [6, 79, 3, 35, 1],
                    [7, 80, 4, 36, 2]])

# row, column
slicing_matrice = matrice[1:3, 1:3]
print(slicing_matrice)
# # of rows by # of elements in row
print(matrice.shape)

# 3D tensors (look like a cube of numbers -- essentially matrices that have been packed into an array!)
tensor = np.array(
    [[[3, 5, 7, 10, 12],
      [3, 5, 6, 4, 5]],
     [[9, 45, 34, 23, 54],
      [3, 5, 7, 31, 57]],
     [[23, 54, 74, 24, 11],
      [9, 34, 65, 34, 34]]])

# of matrices by number of rows in each matrice by number of elements in each row
print(tensor.shape)
slicing_tensor = tensor[:3, 1:, 2:]
print(slicing_tensor)

"""
Element-wise manipulation
"""
# iterates through matrice rows, iterates through element in matrice row, assigns matrice
# element new matrice element of 0 if element < 0
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

print(matrice.shape)
print("***" * 5)
print(matrice[0, 0])
print(naive_relu(matrice))

print("***")
print(naive_add(matrice, matrice2))
print("***")

# Numpy supports faster, parallel tensor manipulations
z = matrice + matrice2 # element wise addition
print(z)

z = np.maximum(z, 0) # element wise relu
print(z)

"""
Broadcasting

Tensor broadcasting is about bringing the tensors of different dimensions / shape 
to the compatible shape such that arithmetic operations can be performed on them.

In broadcasting, the smaller array is found, the new axes are added as per the larger 
array and data is added appropriately to the transformed array.

Code examples from: https://deeplizard.com/learn/video/6_33ulFDuCg
"""

# Example 1: Same shape tensors
tensor1 = np.array([[1, 2, 3],])
tensor2 = np.array([[4, 5, 6],])

tensor3 = tensor1 + tensor2
# expected: [[5 7 9]], shape and rank will be same as tensor 1 and 2
print(tensor3)

# Example 2: Same rank, different shape
# Shape: (1, 3)
tensor4 = np.array([[1, 2, 3],])
# Shape: (3, 1)
tensor5 = np.array(
 [[4],
 [5],
 [6]])

print("----")
print(tensor4.shape)
print(tensor5.shape)
tensor6 = tensor4 + tensor5
"""
What happens in broadcasting? Working from the last dimension, the greatest dimension
becomes the new dimension of the tensor; so in this case (1, 3) and (3, 1) becomes 
(3, 3)

TENSOR 1
Before:
    [[1, 2, 3],]

After:
    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]

TENSOR2 

Before:
    [[4],
     [5],
     [6]]

After:
    [[4, 4, 4],
     [5, 5, 5],
     [6, 6, 6]]
     
Then: the problem becomes: 
    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
+
    [[4, 4, 4],
     [5, 5, 5],
     [6, 6, 6]]
-------------------- 
    [[5, 6, 7],
     [6, 7, 8],
     [7, 8, 9]]  
"""

# Example 3: Different Ranks
tensor7 = np.array([[1, 2, 3]]) # Shape: (1, 3)
tensor8 = np.array(5) # Shape: ()

"""
Here, tensor8 has no shape, so we sub in for a shape of (1,1)
this means that if the new shape is 1, it's compatable. (1, 1)
+ (1, 3) takes the shape of (1, 3)

Before:
    5

After:
    [[5, 5, 5],]

+ [[1, 2, 3],]
= [[6, 7, 8]]
"""
tensor9 = tensor7 + tensor8
print(tensor9)

# rank: 3
# shape: (1,2,3)
tensor10 = np.array(
  [[[1, 2, 3],
  [4, 5, 6]]])


# rank: 2
# shape: (3,3)
tensor11 = np.array(
 [[1, 1, 1],
 [2, 2, 2],
 [3, 3, 3]])

# Illegal tensor operation
#tensor12 = tensor10 + tensor11

tensor12 = np.array(
  [[[1, 2, 3],
  [4, 5, 6]],
  [[5, 7, 9],
  [2, 4, 6]]])

# (2, 2, 3)
print(tensor12.shape)

vector1 = np.array([1,
                    2,
                    3,
                    4])

vector2 = np.array([1,
                    2,
                    3,
                    4])

print(vector1.shape)
vector3 = np.dot(vector1, vector2)
print(vector3)