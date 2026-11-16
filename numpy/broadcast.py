# Broadcasting allows NUMPY to perfom operations on arrays
# With different shapes by virtually expanding dimensions
# So they match the large array's shape.

# The dimensions have the same size.
# Or
# One of the dimensions has a size of 1.

import numpy as np

array1 = np.array([[1, 2 , 3, 4], [5, 6, 7 , 8]])

array2 = np.array([[3],[4],[5],[6]])

print(array1.shape)
#(2, 4)
print(array2.shape)
# 
print(array1 * array2)
