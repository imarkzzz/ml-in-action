from keras.datasets import mnist

# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5 # minmax normalized
x_test = x_test.astype('float32') / 255. - 0.5 # minmax normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))


def main():
  print(x_train.shape)
  print(x_test.shape)

if __name__ == '__main__':
  main()