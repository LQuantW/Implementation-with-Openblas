import numpy as np
import time
import os
import multiprocessing
import random

MATRIX_FILE = 'matrix.npy'
MATRIX_SHAPE = (10000, 1000)

def generate_matrix():
    return np.random.randn(*MATRIX_SHAPE)

def update_matrix(matrix):
    num_elements = np.prod(MATRIX_SHAPE)
    indices = np.random.choice(num_elements, num_elements // 2, replace=False)
    matrix.flat[indices] = np.random.randn(num_elements // 2)
    return matrix

def save_matrix(matrix):
    np.save(MATRIX_FILE, matrix)

def load_matrix():
    return np.load(MATRIX_FILE)

def matrix_updater():
    if os.path.exists(MATRIX_FILE):
        matrix = load_matrix()
    else:
        matrix = generate_matrix()
        save_matrix(matrix)

    while True:
        time.sleep(random.randint(1, 5))
        matrix = update_matrix(matrix)
        save_matrix(matrix)
        print("Matrix updated")

if __name__ == "__main__":
    updater_process = multiprocessing.Process(target=matrix_updater)
    updater_process.start()
    updater_process.join()