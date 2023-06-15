import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image


def k_approx(k, u, s, vh):
    s_new = [0] * s.size
    for i in range(0, k):
        s_new[i] = s[i]

    matrix = u @ np.diag(s_new) @ vh

    return matrix


def visualize_image(matrix):
    im = Image.fromarray(matrix)
    im.show()


def main():
    image = misc.ascent()
    u, s, vh = np.linalg.svd(image)

    visualize_image(k_approx(5, u, s, vh))
    visualize_image(k_approx(20, u, s, vh))
    visualize_image(k_approx(75, u, s, vh))


if __name__ == "__main__":
    main()
