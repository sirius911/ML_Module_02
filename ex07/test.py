import numpy as np
from polynomial_model import add_polynomial_features

x = np.arange(1,6).reshape(-1, 1)
# Example 0:
print(add_polynomial_features(x, 3))
# Output:
# array([[ 1, 1, 1],[ 2, 4, 8],[ 3, 9, 27],[ 4, 16, 64],[ 5, 25, 125]])

# Example 1:
print(add_polynomial_features(x, 6))
# Output:
# array([[ 1, 1, 1, 1, 1, 1],[ 2, 4, 8, 16, 32, 64],[ 3, 9, 27, 81, 243, 729],
#    [ 4, 16, 64, 256, 1024, 4096],[ 5, 25, 125, 625, 3125, 15625]])
