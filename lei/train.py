from model import convolutional_model



def main():
    input_shape = (256,256,3)
    model = convolutional_model(input_shape)
