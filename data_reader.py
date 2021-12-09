import numpy as np
import pickle

def load_training():
    max_pixels = 0

    with open('data/train.pkl', 'rb') as f:
        images = pickle.load(f)
        processed_imgs = []
        for img in images:
            processed_imgs += [[np.append(img[0], [1]), img[1]]]
            if len(img[0]) > max_pixels:
                max_pixels = len(img[0])
                
        return max_pixels + 1, processed_imgs
    
def load_test():
    with open('data/test.pkl', 'rb') as f:
        processed_imgs = []
        for img in pickle.load(f):
            processed_imgs += [[np.append(img[0], [1]), img[1]]]
        return processed_imgs
    
def load_weights():
    with open('output/weights.pkl', 'rb') as f:
        return pickle.load(f)