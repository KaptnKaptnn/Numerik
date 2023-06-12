import numpy as np


def recursive_calc(m, n):
    a = 0

    if m <= n + 1:
        a = 1 / (m + n - 1)
    else:
        a = 0

    return a


def assemble_matrix(size):
    matrix = np.zeros(shape=(size, size))

    for i in range(1, size):
        for j in range(1, size):
            matrix[i][j] = recursive_calc(i, j)

    return matrix


def assemble_vector(size):
    vector = np.zeros(shape=(size))

    for i in range(1, size):
        result = 0

        for n in range(1, size):
            result += recursive_calc(i, n)

        vector[i] = result

    return vector


def qr_decomp(matrix):
    q, r = np.linalg.qr(matrix)
    return q, r


def qr_back_substitution(q, r, b):

    qTb = np.matmul(q.T, b)

    x = np.linalg.solve(r, qTb)

    return x


def main():
    size = 100

    matrix = assemble_matrix(size)
    q, r = qr_decomp(matrix)

    b = assemble_vector(size)

    x = qr_back_substitution(q, r, b)

    print(x)


if __name__ == "__main__":
    main()
