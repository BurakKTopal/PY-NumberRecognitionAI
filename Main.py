import time
from PIL import Image
from Data import get_mnist_train, get_mnist_test, parsingPictureWithoutInversion, resizeImage, separatingNumbers, filterImages
from NeuralNetwork import *
from Helper import plotting

network = neuralNetwork()


def train():
    """"
    Training the neural network with the train set of the MNIST data set
    """

    images_train, labels_train = get_mnist_train()
    print("IMAGES UPLOADED!")

    EPOCH = 5  # Number of epoch's/iterations

    for i in range(EPOCH):
        start_time = time.perf_counter()
        correct = 0
        for image, label in zip(images_train, labels_train):
            image.shape += (1,)
            label.shape += (1,)
            network.forward(image, label)
            if np.argmax(network.output) == np.argmax(label):
                correct += 1
            network.backward()
        end_time = time.perf_counter()
        print(f"accuracy is {correct/len(images_train)}")
        print(f"time spent:{end_time - start_time}")
        network.learning_rate /= 2  # Dividing learning rate after each epoch
        network.save()  # saving the model
    return

def test():
    """"
    Testing the trained neural network with the test set of the MNIST data set
    """

    images_test, labels_test = get_mnist_test()
    print("IMAGES UPLOADED!")
    network.load('modelDEF.pkl')  # Loading model

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


def testMIPerCase(filename):
    """"
    We can use the AI for uploading own pictures and test for results
    """
    list_nums = []
    list_probs = []
    certainty = 1
    guess = ""
    im = Image.open(filename)  # Opening the testing image
    filtered_images = filterImages(im)
    #filtered_images.show()
    splitted_images = separatingNumbers(filtered_images)
    for im in splitted_images:
        im = resizeImage(im, test=True)
        #im.show()
        im.save('static/normalizedPictures/picture.png')
        im = parsingPictureWithoutInversion(im, True)
        #im = im.reshape((28, 28))
        #image = Image.fromarray(np.uint8(im*255))
        #image.show()
        #pixelToPicture(im)
        im.shape = (784, 1)
        network.load('modelDEF.pkl')
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
        #print(str(np.argmax(network.output)))
        guess = guess + str(np.argmax(network.output))

    #print("Certainty(%):", round(certainty*100, 1))
    #print('the guess is:', int(guess))
    plotting(list_nums, list_probs)

    return int(guess), round(certainty*100, 1)


if __name__ == "__main__":
    #testPerCase()
    #print(parsingPicture(im))
    #train()
    #test()
    #testPerCase()
    testMIPerCase('webApplication/static/uploads/picture.png')