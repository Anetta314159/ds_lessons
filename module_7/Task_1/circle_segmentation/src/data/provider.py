
import numpy as np
import cv2

def create_image_and_label(nx, ny, cnt=10, r_min=5, r_max=50, border=10, sigma=20):
    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny), dtype=bool)
    mask = np.zeros((nx, ny), dtype=bool)

    for _ in range(cnt):
        a = np.random.randint(border, nx - border)
        b = np.random.randint(border, ny - border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1, 255)

        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)
        image[m] = h

    label[mask] = 1
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    return image, label
