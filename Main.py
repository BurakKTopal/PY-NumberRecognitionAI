import time
from PIL import Image
from ImageParsing.Data import get_mnist_train, get_mnist_test, parsingPictureWithoutInversion, resizeImage, separatingNumbers, filterImages
from NeuralNetworks.NeuralNetwork4Layers import neuralNetwork4Layers
from Visualisation.Helper import plotting
from Visualisation.HelperLoss import plottingLoss
import numpy as np

network = neuralNetwork4Layers()


def train():
    """"
    Training the neural network with the train set of the MNIST data set
    """
    images_train, labels_train = get_mnist_train()
    print("IMAGES UPLOADED!")
    loss_list = list()
    mean_loss_list = list()
    EPOCH = 5  # Number of epoch's/iterations

    for i in range(EPOCH):
        k = 0
        start_time = time.perf_counter()
        correct = 0
        for image, label in zip(images_train, labels_train):
            image.shape += (1,)
            label.shape += (1,)
            network.forward(image, label)
            loss_list.append(network.loss[0])
            k += 1
            if k == 100:
                mean_loss_list.append(np.mean(loss_list))
                k = 0

            if np.argmax(network.output) == np.argmax(label):
                correct += 1
            network.backward()

        end_time = time.perf_counter()
        print(f"accuracy is {correct/len(images_train)}")
        print(f"time spent:{end_time - start_time}")
        network.learning_rate /= 2  # Dividing learning rate after each epoch
        network.save()  # saving the model
        plottingLoss(mean_loss_list)
    return


def test():
    """"
    Testing the trained neural network with the test set of the MNIST data set
    """

    images_test, labels_test = get_mnist_test()
    print("IMAGES UPLOADED!")
    try:
        network.load('model4Layers.pkl')
    except:
        network.load('SavedModels/model4Layers.pkl')

    start_time = time.perf_counter()
    correct = 0

    for image, label in zip(images_test, labels_test):
        image.shape += (1,)
        label.shape += (1,)
        network.forward(image, label)
        if np.argmax(network.output) == np.argmax(label):
            correct += 1

    end_time = time.perf_counter()
    print(f"accuracy is {correct / len(images_test)}")
    print(f"time spent:{end_time - start_time}")
    return


def ForwardingPictureToNN(filename):
    """"
    We can use the AI for uploading own pictures and test for results
    """
    list_nums = []
    list_probs = []
    certainty = 1
    guess = ""
    im = Image.open(filename)  # Opening the testing image
    im = im.convert("L")
    filtered_images = filterImages(im)
    splitted_images = separatingNumbers(filtered_images)


    for im in splitted_images:
        im = resizeImage(im, test=True)
        try:  # If you run this file, the wd doesn't match with the given path name
            im.save('static/normalizedPictures/picture.png')
        except Exception:
            pass
        im = parsingPictureWithoutInversion(im, True)
        im.shape = (784, 1)
        try:
            network.load('model4Layers.pkl')
        except:
            network.load('SavedModels/model4Layers.pkl')
        network.forward(im, np.empty(1))  # As there is no target in this case, an empty matrix will be sent as target

        output = np.array(network.output)
        certainty = certainty*round(network.output[np.argmax(network.output)][0] / (sum(network.output))[0], 3)

        output = output.reshape(1, 10)[0]

        DIMENSION = 3  # Taking the 'DIMENSION' number of best guesses of the network
        top_three_numbers = np.argpartition(output, -DIMENSION)[-DIMENSION:].tolist()  # Taking the best guesses
        top_three_numbers_probs = output[top_three_numbers].tolist()

        for index in range(len(top_three_numbers)):
            top_three_numbers[index] = str(top_three_numbers[index])  # List must have string as elements, for the tensor
                                                                        # product in data.py

        for index in range(len(top_three_numbers_probs)):  # Calculating the certainty(value/sum)
            top_three_numbers_probs[index] = round(top_three_numbers_probs[index] / (sum(output)), 3)

        list_nums.append(top_three_numbers)
        list_probs.append(top_three_numbers_probs)
        guess = guess + str(np.argmax(network.output))

    plotting(list_nums, list_probs)

    return int(guess), round(certainty*100, 1)

if __name__ == "__main__":
    #train()
    test()