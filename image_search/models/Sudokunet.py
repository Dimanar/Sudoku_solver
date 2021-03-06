from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Sequential

class SudokuNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(32, (5, 5), padding='same',
                  input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
