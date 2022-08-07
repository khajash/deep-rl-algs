
def get_conv2d_out_dim(size, kernel, stride=1, padding=0):
    return (size + 2*padding - kernel)//stride + 1