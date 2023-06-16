import numpy as np
import time
import matplotlib.pyplot as plt

def timing_decorator(func):
    execution_times = []
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        return result

    wrapper.execution_times = execution_times
    return wrapper

@timing_decorator
def assemble_matrix(size):
    matrix = np.zeros(shape=(size, size))

    for i in range(1, size):
        for j in range(1, size):

            if(i <= j + 1):
                matrix[i][j] = 1 / (i + j - 1)
            else:
                matrix[i][j] = 0

    return matrix

@timing_decorator
def assemble_vector(size):
    vector = np.zeros(shape=(size))

    for i in range(1, size):
        result = 0

        for n in range(1, size):

            if(i <= n + 1):
                result += 1 / (i + n - 1)

        vector[i] = result

    return vector

@timing_decorator
def qr_decomp(matrix):
    q, r = np.linalg.qr(matrix)
    return q, r

@timing_decorator
def qr_back_substitution(q, r, b):

    qTb = np.matmul(q.T, b)
    size = qTb.size
    x = np.zeros(size)

    for n in range(size - 1, 0, -1):
        x[n] = (qTb[n] - np.dot(r[n, n:size], x[n:size])) / r[n,n]

    return x

def qr_calc(size):

    time = 0
    exec_run = 0

    matrix = assemble_matrix(size)
    exec_run = len(assemble_matrix.execution_times) - 1
    time += assemble_matrix.execution_times[exec_run]

    q, r = qr_decomp(matrix)
    time += qr_decomp.execution_times[exec_run]

    b = assemble_vector(size)
    time += assemble_vector.execution_times[exec_run]

    x = qr_back_substitution(q, r, b)
    time += qr_back_substitution.execution_times[exec_run]

    return x, time

def main():

    qr_calc(300)

    n_values = np.arange(100, 10001, 50)
    n_exec_time = np.zeros(n_values.size)

    for n in range(0, n_values.size):
        matrix, time = qr_calc(n_values[n])
        print(n_values[n])
        n_exec_time[n] = time
    

    print(n_values)
    print(n_exec_time)

    plt.plot(n_values, n_exec_time)
    plt.ylabel('Execution Time')
    plt.xlabel('Size of N')
    plt.savefig('Aufgabe1_ExecutionTimePlot_Kaloyan_Kondov_2360001_Marlon_Schnell_2307973.png')
    plt.show()


if __name__ == "__main__":
    main()
