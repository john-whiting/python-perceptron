import numpy as np
import random
import pickle
from data_reader import load_training

class Perceptron:
    def __init__(self, size, U):
        random.seed()
        self.u = U
        self.w = np.array([random.randint(0, 100) for i in range(size)])
        self.converged = False
        self.iter = 0
    
    def _process(self, s):
        x, p = s
        d = np.dot(self.w, x)
        
        if p == 1 and d < 0:
            # P should be above 0
            return False, 1
        if p == 0 and d >= 0:
            # N should be below 0
            return False, -1
        
        return True, 0

    def update_converged(self):
        for s in self.u:
            if not self._process(s)[0]:
                return 
        self.converged = True

    def process(self):
        while not self.converged:
            s = self.u[random.randint(0, len(self.u) - 1)]
            x, _ = s
            match, pm = self._process(s)
            
            if not match:
                self.w = self.w + (pm * x)
            
            self.update_converged()
            self.iter += 1
    
# Load the data
print('Processing data...')
perc = Perceptron(*load_training())
perc.process()

# Save the weights for testing
f = open('output/weights.pkl', 'wb')
pickle.dump(perc.w, f)
f.close()
print(f'Finished processing data in {perc.iter} iterations.\nWeights are stored in weights.pkl')