import numpy as np
import os
import fcntl
import multiprocessing
from datetime import datetime
import psutil

MATRIX_FILE = 'matrix.npy'
CORR_FILE = 'correlation.npy'
LOG_FILE = 'calculation_log.txt'
LOCK_FILE = 'calculation.lock'
CHUNK_SIZE = 100  # 每次处理的列数

def log_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f'{timestamp} - {message}\n')

def load_matrix(file_path):
    with open(file_path, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        matrix = np.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return matrix

def save_correlation(correlation, file_path):
    with open(file_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        np.save(f, correlation)
        fcntl.flock(f, fcntl.LOCK_UN)

def calculate_correlation_chunk(matrix, start_col, end_col, return_dict, process_index):
    rows, cols = matrix.shape
    mean = np.mean(matrix, axis=0)
    stddev = np.std(matrix, axis=0)
    
    corr_matrix_chunk = np.empty((end_col - start_col, cols))
    for i in range(start_col, end_col):
        for j in range(cols):
            if i == j:
                corr_matrix_chunk[i - start_col, j] = 1.0
            else:
                covariance = np.mean((matrix[:, i] - mean[i]) * (matrix[:, j] - mean[j]))
                corr_matrix_chunk[i - start_col, j] = covariance / (stddev[i] * stddev[j])
    
    return_dict[process_index] = corr_matrix_chunk

def correlation_calculator():
    if not os.path.exists(MATRIX_FILE):
        log_message("Matrix file does not exist.")
        return

    matrix = load_matrix(MATRIX_FILE)
    rows, cols = matrix.shape
    
    # 获取系统逻辑处理器数量的一半
    total_logical_processors = psutil.cpu_count(logical=True)
    max_workers = total_logical_processors // 2
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    
    # 分块处理矩阵
    for start_col in range(0, cols, CHUNK_SIZE):
        end_col = min(start_col + CHUNK_SIZE, cols)
        process = multiprocessing.Process(target=calculate_correlation_chunk, 
                                          args=(matrix, start_col, end_col, return_dict, start_col // CHUNK_SIZE))
        processes.append(process)
        
        # 限制同时运行的进程数量
        if len(processes) == max_workers:
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            processes = []

    # 处理剩余的进程
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # 合并结果
    correlation_matrix = np.empty((cols, cols))
    for key, chunk in return_dict.items():
        start_col = key * CHUNK_SIZE
        end_col = start_col + chunk.shape[0]
        correlation_matrix[start_col:end_col, :] = chunk

    save_correlation(correlation_matrix, CORR_FILE)
    log_message("Correlation matrix calculated and saved.")
    print("Correlation matrix calculated and saved.")

if __name__ == "__main__":
    correlation_calculator()