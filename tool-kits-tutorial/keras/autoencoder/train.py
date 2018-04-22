from brain import AutoencoderNet
import matplotlib.pyplot as plt
from env import x_train, x_test, y_test


def train():
  autoencoder, encoder = AutoencoderNet()
  # compile autoencoder
  autoencoder.compile(optimizer='adam', loss='mse')

  # training
  autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)

  # plotting
  encoded_imgs = encoder.predict(x_test)
  plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
  plt.colorbar()
  plt.show()
  
  autoencoder.save('autoencoder.h5')
  encoder.save('encoder.h5')

  autoencoder.save_weights('autoencoder_weights.h5')
  encoder.save_weights('encoder_weights.h5')

  # save and load fresh network without trained weights
  from keras.models import model_from_json
  autoencoder_json = autoencoder.to_json()
  with open('autoencoder.json', 'w') as fout:
    fout.write(autoencoder_json)
  
  encoder_json = encoder.to_json()
  with open('encoder.json', 'w') as fout:
    fout.write(encoder_json)


def main():
  train()

if __name__ == '__main__':
  main()