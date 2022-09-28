# Application 1 - Step 1 - Import the dependencies
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.optimizers import gradient_descent_v2
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import layers
from matplotlib import pyplot
import cv2


#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
def summarizeLearningCurvesPerformances(histories, accuracyScores):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='green', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='red', label='test')

        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='green', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='red', label='test')

        # print accuracy for each split
        print("Accuracy for set {} = {}".format(i, accuracyScores[i]))

    pyplot.show()

    print('Accuracy: mean = {:.3f} std = {:.3f}, n = {}'.format(np.mean(accuracyScores) * 100,
                                                                np.std(accuracyScores) * 100, len(accuracyScores)))


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def prepareData(trainX, trainY, testX, testY):
    # TODO - Application 1 - Step 4a - reshape the data to be of size [samples][width][height][channels]
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)).astype('float32')
    testX = testX.reshape((testX.shape[0], testX.shape[1], testX.shape[2], 1)).astype('float32')

    # TODO - Application 1 - Step 4b - normalize the input values
    trainX = trainX / 255
    testX = testX / 255

    # TODO - Application 1 - Step 4c - Transform the classes labels into a binary matrix
    trainY = np_utils.to_categorical(trainY)
    testY = np_utils.to_categorical(testY)
    num_classes = testY.shape[1]

    return trainX, trainY, testX, testY


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineModel(input_shape, num_classes):
    # TODO - Application 1 - Step 6a - Initialize the sequential model
    model = keras.models.Sequential()

    # TODO - Application 1 - Step 6b - Create the first hidden layer as a convolutional layer
    model.add(layers.Convolution2D(64, input_shape=(28, 28, 1), kernel_initializer='he_uniform', kernel_size=3,
                                   activation='relu'))

    # TODO - Application 1 - Step 6c - Define the pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # TODO - Application 1 - Step 6d - Define the flatten layer
    model.add(layers.Flatten())

    # TODO - Application 1 - Step 6e - Define a dense layer of size 16
    model.add(layers.Dense(16, activation='relu'))

    # TODO - Application 1 - Step 6f - Define the output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # TODO - Application 1 - Step 6g - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=gradient_descent_v2.SGD(learning_rate=0.01, momentum=0.9),
                  metrics=['accuracy'])

    return model


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY):
    # TODO - Application 1 - Step 6 - Call the defineModel function
    input_shape = [trainX.shape[1], trainX.shape[2], 1]
    num_classes = testY.shape[1]
    predictedLabel = defineModel(input_shape, num_classes)

    # TODO - Application 1 - Step 7 - Train the model
    predictedLabel.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, batch_size=32, verbose=2)
    predictedLabel.save('Fashion_MNIST_model.h5')
    # TODO - Application 1 - Step 8 - Evaluate the model
    scores = predictedLabel.evaluate(testX, testY, verbose=0)
    print("error : {:.2f}".format(100 - scores[1] * 100))
    return


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY):
    k_folds = 5

    accuracyScores = []
    histories = []

    # Application 2 - Step 2 - Prepare the cross validation datasets
    kfold = KFold(k_folds, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(trainX):
        # TODO - Application 2 - Step 3 - Select data for train and validation
        trainX_i = trainX[train_idx]
        trainY_i = trainY[train_idx]
        valX_i = trainX[val_idx]
        valY_i = trainY[val_idx]

        # TODO - Application 2 - Step 4 - Build the model - Call the defineModel function
        model = defineModel((28, 28, 1), 10)

        # TODO - Application 2 - Step 5 - Fit the model
        history = model.fit(trainX_i, trainY_i, epochs=5, batch_size=32, validation_data=(valX_i, valY_i), verbose=1)

        # TODO - Application 2 - Step 6 - Save the training related information in the histories list
        histories.append(history)

        # TODO - Application 2 - Step 7 - Evaluate the model on the test dataset
        scores = model.evaluate(testX, testY, verbose=0)

        # TODO - Application 2 - Step 8 - Save the accuracy in the accuracyScores list
        accuracyScores.append(scores[1])

    return histories, accuracyScores


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():
    # TODO - Application 1 - Step 2 - Load the Fashion MNIST dataset in Keras
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    for i in range(9):
        cv2.imshow('Image', trainX[i])

    # TODO - Application 1 - Step 3 - Print the size of the train/test dataset
    print("train x size: {}".format(len(trainX)))
    print("train y size: {}".format(len(trainY)))

    print("test x size: {}".format(len(testX)))
    print("test y size: {}".format(len(testY)))

    # TODO - Application 1 - Step 4 - Call the prepareData method
    trainX, trainY, testX, testY = prepareData(trainX, trainY, testX, testY)

    # TODO - Application 1 - Step 5 - Define, train and evaluate the model in the classical way
    defineTrainAndEvaluateClassic(trainX, trainY, testX, testY)

    # TODO - Application 2 - Step 1 - Define, train and evaluate the model using K-Folds strategy
    # histories, scores = defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY)

    # TODO - Application 2 - Step9 - System performance presentation
    # summarizeLearningCurvesPerformances(histories, scores)

    return


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
