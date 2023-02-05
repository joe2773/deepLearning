import numpy as np
from numpy import ndarray
class Assert():
    def same_shape(self,array_1: ndarray, array_2: ndarray) -> None:
        if(array_1.size != array_2.size):
            raise ValueError("Arrays must be same length")