
# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np


def generate_synthetic_data(num_points=20):
    
    x = np.random.uniform(-4, 4, num_points)
    
    y = x**3 + np.random.normal(0, 3**2, num_points)
    
    return x, y
    


