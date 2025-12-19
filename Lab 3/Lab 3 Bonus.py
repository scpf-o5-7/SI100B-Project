import numpy as np
import cv2


def conv2d_batch(x, kernel, stride=1, padding=0):
    if len(x.shape) == 4:
        B, H, W, C_in = x.shape
    else:
        B, H, W, C_in = 1, x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(1, H, W, C_in)

    N_filters, Kh, Kw, _ = kernel.shape

    Ho = (H - Kh) // stride + 1
    Wo = (W - Kw) // stride + 1

    output = np.zeros((B, Ho, Wo, N_filters))

    for b in range(B):
        for f in range(N_filters):
            for i in range(Ho):
                for j in range(Wo):
                    h_start = i * stride
                    h_end = h_start + Kh
                    w_start = j * stride
                    w_end = w_start + Kw

                    region = x[b, h_start:h_end, w_start:w_end, :]
                    output[b, i, j, f] = np.sum(region * kernel[f])

    return output if B > 1 else output[0]


def ReLU(x):
    return np.maximum(0, x)


def MaxPool(x, size=2):
    if len(x.shape) == 4:
        B, H, W, C = x.shape
        stride = size
        Ho = (H - size) // stride + 1
        Wo = (W - size) // stride + 1

        output = np.zeros((B, Ho, Wo, C))

        for b in range(B):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        h_start = i * stride
                        h_end = h_start + size
                        w_start = j * stride
                        w_end = w_start + size

                        window = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, i, j, c] = np.max(window)
        return output

    elif len(x.shape) == 3:
        H, W, C = x.shape
        stride = size
        Ho = (H - size) // stride + 1
        Wo = (W - size) // stride + 1

        output = np.zeros((Ho, Wo, C))

        for c in range(C):
            for i in range(Ho):
                for j in range(Wo):
                    h_start = i * stride
                    h_end = h_start + size
                    w_start = j * stride
                    w_end = w_start + size

                    window = x[h_start:h_end, w_start:w_end, c]
                    output[i, j, c] = np.max(window)
        return output


if __name__ == "__main__":
    image_files = ["img/sad.jpg", "img/sad1.jpg", "img/sad2.jpg", "img/sad3.jpg"]
    images = []

    for file in image_files:
        img = cv2.imread(file)
        img = img.astype(np.float32) / 127.0 - 1.0
        images.append(img)

    input_image = np.array(images)
    print("input shape:", input_image.shape)

    conv_w = np.load("conv_w.npy")
    fc_w = np.load("fc.npy")

    print("conv weight shape:", conv_w.shape)
    print("fc weight shape:", fc_w.shape)

    out = conv2d_batch(input_image, conv_w, stride=1, padding=0)
    print("after conv:", out.shape)

    out = ReLU(out)
    print("after relu:", out.shape)

    out = MaxPool(out, size=2)
    print("after pool:", out.shape)

    B = out.shape[0]
    out = out.reshape(B, -1)
    print("after flatten:", out.shape)

    if out.shape[1] != fc_w.shape[0]:
        print(f"dim mismatch: {out.shape[1]} vs {fc_w.shape[0]}")
        if out.shape[1] > fc_w.shape[0]:
            out = out[:, : fc_w.shape[0]]
        else:
            pad_width = fc_w.shape[0] - out.shape[1]
            out = np.pad(out, ((0, 0), (0, pad_width)), mode="constant")

    b = np.zeros(fc_w.shape[1])
    output = np.dot(out, fc_w) + b

    print("final output shape:", output.shape)
    print("\noutput list:")
    print(output.tolist())
