import numpy as np
import cv2

def MaxPool(x, size):
    pass

def ReLU(x):
    pass

if __name__ == '__main__':
    batch_size = 4

    # 1 read image 
    # (Batch_size, Height, Width, Channels)
    input_image = np.array([cv2.imread('sad.jpg'),
                        cv2.imread('sad1.jpg'),
                        cv2.imread('sad2.jpg'),
                        cv2.imread('sad3.jpg')])
    print(input_image.shape)

    # 2 load weights from local 
    conv_w = np.load("conv_w.npy")
    fc_w = np.load("fc.npy")

    # Using maxpool as pooling fucntion and using ReLU as activation function.
    # 3 inference

    #print(output)