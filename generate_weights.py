from typing import List

import numpy as np
import os
import pickle
import random
import time

from data_reader import load_training

class Perceptron:
    u: List[List[np.ndarray | int]] = None
    """
    Represents the universal set of data. Each member of the list is another list.
    member[0] is the numpy array.
    member[1] is the p value whether or not a given member should be positive or negative.
    """
    w: np.ndarray = None
    """Represents the array of weights of the perceptron"""
    converged = False
    """Represents whether the data has converged or not under the current weight array"""
    iter = 0
    """Represents the total number of iterations of the process loop"""
    
    def __init__(self, U):
        random.seed(time.time() * 1000)
        self.u = U
        self.w = np.array([random.randint(0, 100) for i in range(28 * 28 + 1)])
        self.converged = False
        self.iter = 0
    
    def _process_point(self, point_info):
        # x is the array of data points
        # p is if it is positive or negative (1 = positive, 0 = negative)
        x, p = point_info
        d = np.dot(self.w, x)
        
        if p == 1 and d < 0:
            # P should be above 0
            return False, 1
        if p == 0 and d >= 0:
            # N should be below 0
            return False, -1
        
        return True, 0

    def update_converged(self):
        for point_info in self.u:
            if not self._process_point(point_info)[0]:
                return 
        self.converged = True

    def process(self):
        while not self.converged:
            point_info = self.u[random.randint(0, len(self.u) - 1)]
            x, _ = point_info
            match, pm = self._process_point(point_info)
            
            if not match:
                self.w = self.w + (pm * x)
            
            self.update_converged()
            self.iter += 1
    
# Load the data
print('Processing data...')
perc = Perceptron(load_training())
perc.process()

# Create output folder
if not os.path.exists('output'):
    os.mkdir('output')
    
# Save the weights for testing
f = open('output/weights.pkl', 'wb')
pickle.dump(perc.w, f)
f.close()
print(f'Finished processing data in {perc.iter} iterations.\nWeights are stored in weights.pkl')