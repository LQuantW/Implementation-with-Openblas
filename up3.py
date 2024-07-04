import numpy as np
import time
import os
import random
import multiprocessing
import fcntl
from datetime import datetime

MATRIX_FILE = 'matrix.npy'
TEMP_MATRIX_FILE = 'matrix_temp.npy'
BACKUP_MATRIX_FILE = 'matrix_backup.npy'
LOG_FILE = 'update_log.txt'
LOCK_FILE = 'update.lock'
MATRIX_SHAPE = (10000, 1000)

def log_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f'{timestamp} - {message}\n')

def generate_matrix():
    return np.random.randn(*MATRIX_SHAPE)

def update_matrix(matrix):
    num_elements = np.prod(MATRIX_SHAPE)
    indices = np.random.choice(num_elements, num_elements // 2, replace=False)
    matrix.flat[indices] = np.random.randn(num_elements // 2)
    return matrix

def save_matrix(matrix, file_path):
    with open(file_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        np.save(f, matrix)
        fcntl.flock(f, fcntl.LOCK_UN)

def load_matrix(file_path):
    with open(file_path, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        matrix = np.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return matrix

def acquire_lock(lock_file):
    lock_fd = open(lock_file, 'w')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock_fd

def matrix_updater():
    lock_fd = acquire_lock(LOCK_FILE)
    
    if os.path.exists(MATRIX_FILE):
        matrix = load_matrix(MATRIX_FILE)
    else:
        matrix = generate_matrix()
        save_matrix(matrix, MATRIX_FILE)
        log_message("Initial matrix created and saved.")

    while True:
        time.sleep(random.randint(1, 5))
        
        # 创建备份文件
        save_matrix(matrix, BACKUP_MATRIX_FILE)
        
        # 更新矩阵
        matrix = update_matrix(matrix)
        
        # 保存临时文件
        save_matrix(matrix, TEMP_MATRIX_FILE)
        
        # 原子性替换原始文件
        os.replace(TEMP_MATRIX_FILE, MATRIX_FILE)
        log_message("Matrix updated and saved.")
        print("Matrix updated")

if __name__ == "__main__":
    try:
        updater_process = multiprocessing.Process(target=matrix_updater)
        updater_process.start()
        updater_process.join()
    except BlockingIOError:
        print("Another instance is already running.")
        log_message("Failed to start: another instance is already running.")