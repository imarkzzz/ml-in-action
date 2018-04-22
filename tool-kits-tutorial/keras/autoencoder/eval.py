import matplotlib.pyplot as plt
from keras.models import load_model, model_from_json

from brain import AutoencoderNet
from env import x_test, y_test


def restore_autoencoder():
  autoencoder = load_model('autoencoder.h5')
  encoder = load_model('encoder.h5')
  encoded_imgs = encoder.predict(x_test)
  plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
  plt.colorbar()
  plt.show()

def restore_autoencoder_weights():
  autoencoder, encoder = AutoencoderNet()
  autoencoder.load_weights('autoencoder_weights.h5')
  encoder.load_weights('encoder_weights.h5')
  encoded_imgs = encoder.predict(x_test)
  plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
  plt.colorbar()
  plt.show()

def restore_autoencoder_json():
  with open('autoencoder.json', 'r') as fin:
    autoencoder_json = fin.readline()
  autoencoder = model_from_json(autoencoder_json)
  with open('encoder.json', 'r') as fin:
    encoder_json = fin.readline()
  encoder = model_from_json(encoder_json)
  encoded_imgs = encoder.predict(x_test)
  plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
  plt.colorbar()
  plt.show()

def main():
  # restore_autoencoder()
  # restore_autoencoder_weights()
  restore_autoencoder_json()

if __name__ == '__main__':
  main()
