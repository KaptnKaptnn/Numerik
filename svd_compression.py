import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image


def k_approx(k, u, s, vh):
    for i in range(k, s.size):
        s[i] = 0

    print(s)
    matrix = u @ np.diag(s) @ vh

    return matrix


def main():
    image = misc.ascent()
    u, s, vh = np.linalg.svd(image)

    print(s)
    im = Image.fromarray(k_approx(75, u, s, vh))
    # im = Image.fromarray(image)
    im.show()


if __name__ == "__main__":
    main()
