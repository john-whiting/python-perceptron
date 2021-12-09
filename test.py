import numpy as np
from data_reader import load_test, load_weights

tests = load_test()
w = load_weights()

def passed_test(s):
    x, p = s
    d = np.dot(w, x)
    
    if p == 1 and d < 0:
        # P should be above 0
        return False
    
    if p == 0 and d >= 0:
        # N should be below 0
        return False
    
    return True

count_failed = 0

for test in tests:
    if not passed_test(test):
        count_failed += 1
    
print('Number of failed tests: ', count_failed)