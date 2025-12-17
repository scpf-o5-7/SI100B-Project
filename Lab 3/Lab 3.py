import numpy as np
import cv2


def conv2d(x, kernel, stride=1, padding=0) -> np.array:
    if padding > 0:
        x_padded = np.pad(x, pad_width=padding, mode="constant", constant_values=0)
    else:
        x_padded = x

    input_h, input_w = x_padded.shape
    kernel_h, kernel_w = kernel.shape

    output_h = (input_h - kernel_h) // stride + 1
    output_w = (input_w - kernel_w) // stride + 1

    output = np.zeros((output_h, output_w))

    for i in range(0, output_h):
        for j in range(0, output_w):
            h_start = i * stride
            h_end = h_start + kernel_h
            w_start = j * stride
            w_end = w_start + kernel_w

            window = x_padded[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(window * kernel)

    return output


def sigmod(x) -> np.array:

    return 1 / (1 + np.exp(-x))


def max_pool_2d(x, pool_size=2, stride=2):

    input_h, input_w = x.shape

    output_h = (input_h - pool_size) // stride + 1
    output_w = (input_w - pool_size) // stride + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size

            window = x[h_start:h_end, w_start:w_end]
            output[i, j] = np.max(window)

    return output


def flatten(x) -> np.array:

    return x.flatten()


def full_connect(x, W, b) -> np.array:

    return np.dot(x, W) + b


if __name__ == "__main__":
    # Read test case
    img = cv2.imread("sad.jpg", cv2.IMREAD_GRAYSCALE)
    # Normailize to [-1, 1)
    img = img / 127.0 - 1.0
    # Set parameters
    con_kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
    W = np.array([[0.01] * 529, [0.02] * 529, [0.03] * 529]).T
    b = np.array([1, 1, 1])

    # 1. Convolution
    out = conv2d(img, con_kernel)
    print(f"After Conv2D: {out.shape}")

    # 2. Activation (sigmod)
    out = sigmod(out)
    print(f"After sigmod: {out.shape}")

    # 3. Max Pooling
    out = max_pool_2d(out, pool_size=2, stride=2)
    print(f"After MaxPool2D: {out.shape}")

    # 4. Flatten
    out = flatten(out)
    print(f"After Flatten: {out.shape}")

    # 5. Fully Connected
    prob = full_connect(out, W, b)
    print(f"Final Output Scores: {prob}")
