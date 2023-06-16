import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
import math


def k_approx(k, matrix):
    u, s, vh = np.linalg.svd(matrix)

    s_new = np.zeros(s.size)
    for i in range(0, k):
        s_new[i] = s[i]

    matrix = u @ np.diag(s_new) @ vh

    return matrix


def visualize_image(matrix, name):
    im = Image.fromarray(matrix)
    im = im.convert("L")
    im.show()
    im.save(name)

def calculate_frobenius(k, matrix):
    k_matrix = k_approx(k, matrix)
    
    approx_error = 0
    dimension = matrix.shape

    for i in range(0, dimension[0] - 1):
        for j in range(0, dimension[1] - 1):

            approx_error += (matrix[i][j] - k_matrix[i][j]) ** 2
    
    approx_error = math.sqrt(approx_error)

    return approx_error



def main():
    image = misc.ascent()
    print(np.linalg.matrix_rank(image))
    u, s, vh = np.linalg.svd(image)

    visualize_image(k_approx(5, image), "Aufgabe2_5_approx_Kaloyan_Kondov_2360001_Marlon_Schnell_2307973.png")
    visualize_image(k_approx(20, image), "Aufgabe2_20_approx_Kaloyan_Kondov_2360001_Marlon_Schnell_2307973.png")
    visualize_image(k_approx(75, image), "Aufgabe2_75_approx_Kaloyan_Kondov_2360001_Marlon_Schnell_2307973.png")

    k_values = np.arange(1, np.linalg.matrix_rank(image))
    k_approx_errors = np.zeros(k_values.size)

    for n in range(0, k_values.size - 1):
        k_approx_errors[n] = calculate_frobenius(k_values[n], image)
        print(k_values[n])

    print(k_approx_errors)

    plt.plot(k_values, k_approx_errors)
    plt.xlabel("Values for k")
    plt.ylabel("Approximation Error")
    plt.savefig('Aufgabe2_ApproximationErrorGraph__Kaloyan_Kondov_2360001_Marlon_Schnell_2307973.png')
    plt.show()


if __name__ == "__main__":
    main()
