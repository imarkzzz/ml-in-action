from keras.models import Model
from keras.layers import Dense, Input

def AutoencoderNet():
  # in order to plot in a 2D figure
  encoding_dim = 2

  # this is our input placeholder
  input_img = Input(shape=(784,))

  # encoder layers
  encoded = Dense(128, activation='relu')(input_img)
  encoded = Dense(64, activation='relu')(encoded)
  encoded = Dense(10, activation='relu')(encoded)
  encoder_output = Dense(encoding_dim)(encoded)

  # decoder layers
  decoded = Dense(10, activation='relu')(encoder_output)
  decoded = Dense(64, activation='relu')(decoded)
  decoded = Dense(128, activation='relu')(decoded)
  decoded = Dense(784, activation='tanh')(decoded)

  # construct the autoencoder model
  autoencoder = Model(input=input_img, output=decoded)

  # constrct the encoder model for plotting
  encoder = Model(input=input_img, output=encoder_output)
  return autoencoder, encoder

def main():
  autoencoder, encoder = AutoencoderNet()

if __name__ == '__main__':
  main()