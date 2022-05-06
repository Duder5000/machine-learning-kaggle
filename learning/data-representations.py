import numpy as np

# Scalars (OD tensors)
scalar = np.array(4)

# Vectors (1D tensors)
vector = np.array([1, 3, 6, 19, 4])

# Matrices (2D tensors)
matrice = np.array([[5, 78, 2, 34, 0],
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