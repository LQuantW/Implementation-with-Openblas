import numpy as np
import os
import time
import multiprocessing
import fcntl

MATRIX_FILE = 'matrix.npy'
CORR_FILE = 'correlation.npy'
BACKUP_MATRIX_FILE = 'matrix_backup.npy'

def calculate_correlation_manual(matrix):
    rows, cols = matrix.shape
    mean = np.mean(matrix, axis=0)
    stddev = np.std(matrix, axis=0)
    
    corr_matrix = np.empty((cols, cols))
    for i in range(cols):
        for j in range(cols):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                covariance = np.sum((matrix[:, i] - mean[i]) * (matrix[:, j] - mean[j])) / (rows - 1)
                corr_matrix[i, j] = covariance / (stddev[i] * stddev[j])
                corr_matrix[j, i] = corr_matrix[i, j]
    return corr_matrix

def save_correlation(corr_matrix, file_path):
    with open(file_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        np.save(f, corr_matrix)
        fcntl.flock(f, fcntl.LOCK_UN)

def load_matrix(file_path):
    with open(file_path, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        matrix = np.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return matrix

def correlation_calculator():
    last_mod_time = 0
    while True:
        if os.path.exists(MATRIX_FILE):
            mod_time = os.path.getmtime(MATRIX_FILE)
            if mod_time != last_mod_time:
                last_mod_time = mod_time
                matrix = load_matrix(MATRIX_FILE)
                corr_matrix = calculate_correlation_manual(matrix)
                save_correlation(corr_matrix, CORR_FILE)
                print("Correlation matrix updated")
        time.sleep(1)

if __name__ == "__main__":
    calculator_process = multiprocessing.Process(target=correlation