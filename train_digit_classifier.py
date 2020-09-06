from image_search.models.Sudokunet import SudokuNet
from keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model after training")
args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 10
BS = 128

print('[INFO] accessing MNIST...')
((trainData, trainLabels), (testData, testLabels)) = \
    mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print("[INFO] training network...")
history = model.fit(
    trainData, trainLabels,
    validation_data=(testData, testLabels),
    batch_size=BS,
    epochs=EPOCHS,
    verbose=1
)

fig1 = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# summarize history for loss
fig2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("[INFO] evaluating network...")
predict = model.predict(testData)
print(classification_report(testLabels.argmax(axis=1),
                            predict.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")