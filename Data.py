import numpy as np
import pathlib
from ImageManipulations import *


def get_mnist_train():
    """""
    Getting the data set from which the model is being trained
    """
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]  # Uploading train images

    for index in range(len(images)):
        image = images[index]
        image = Image.fromarray(np.uint8(image))

        image = resizeImage(image, False)  # Normalizing the pictures of the MNIST data set
        adapted_image = parsingPictureWithoutInversion(image, False)  # Adding the image manipulations to the mnist set
        images[index] = np.reshape(adapted_image, (28, 28))  # Saving normalized picture

    images = images.astype("float32") / 255  # Pixel brightness value as input elements
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]

    return images, labels  # returning array with images and corresponding labels


def get_mnist_test():
    """""
    Getting the data set from which the model is being trained
    """
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_test"], f["y_test"]  # Uploading test data set

    for index in range(len(images)):
        image = images[index]
        image = Image.fromarray(np.uint8(image)) # Creating image from array of pixel brightness
        image = resizeImage(image, False)   # normalizing the picture
        adapted_image = parsingPictureWithoutInversion(image, False)  # Adding the image manipulations to the mnist set
        images[index] = np.reshape(adapted_image, (28, 28))

    images = images.astype("float32") / 255 # saving the entries of the image arrays as float with values between 0-1

    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels  # returning array with images and corresponding labels



def parsingPictureWithoutInversion(im, test):
    """"
    Parsing image to array of pixel brightness values
    """

    if not test:
        im = noise(im) # Only adding a noise factor to the train set to simulate real-world error

    width, height = 28, 28  # Height and width of image(28px by 28px)
    #conversion = np.zeros(784)

    if test:
        im = im.resize((width, height))  # Self-manufactured pictures must be resized.

    im = im.convert("L")  # converting to gray to get the picture brightness

    conversion = np.array(im) / 255

    return conversion  # returning list of darkness scheme of the picture


def parsingPicture(im, test):
    """"
    Parsing image to array of pixel brightness values
    """

    if im.size[0] < 28:  # To avoid conversion issues, we first would like to have all pictures a size bigger then 28x28
        im = ImageOps.invert(im)  # ASSUMING THAT BACKGROUND IS WHITE
        im = im.convert("L")  # Converting to gray to get pixel brightness

        image_array = np.array(im) # Conversion image -> array of pixel brightness
        image_array = np.transpose(image_array) # It is easier to identify null rows if we transpose the matrix first

        image_array = image_array.tolist()  # 'insert( , )' and 'append()'functions possible for lists
        null_row = np.array([0 for _ in range(im.size[1])])
        for i in range(28-im.size[0]//2 + 1):
            image_array.insert(0, null_row)  # Adding 15 null rows to right and left site of original picture
        for i in range(28-im.size[0]//2 + 1):
            image_array.append(null_row)

        image_array = np.array(image_array)  # Converting list back to array
        image_array = np.transpose(image_array)  # Undoing the transposing (transpose(transpose(A)) = A for matrix A)
        im = Image.fromarray(np.uint8(image_array))  # Creating image back
        im = ImageOps.invert(im)  # Undoing the inverting

    elif im.size[0] > 60:
        im = im.convert("L")
        image_array = np.transpose(np.array(im))
        non_white_rows = np.where(image_array.min(axis=1) < 255)  # Find the top and bottom boundaries of the non-white area

        top_boundary = max(0,non_white_rows[0][0] - 2)  # Centering number 2 rows from edge of image
        bottom_boundary = min(im.size[0], non_white_rows[0][-1] + 2)
        cropped_image_array = image_array[top_boundary:bottom_boundary + 1, :]
        cropped_image_array = np.transpose(cropped_image_array) # transposing back
        im = Image.fromarray(np.uint8(cropped_image_array))


    if im.size[1] > 60:
        im = im.convert("L")
        image_array = np.array(im)
        non_white_rows = np.where(image_array.min(axis=1) < 255)  # Find the top and bottom boundaries of the non-white area

        top_boundary = max(0, non_white_rows[0][0] - 2)  # Centering number 2 rows from edge of image
        bottom_boundary = min(im.size[1], non_white_rows[0][-1] + 2)


        cropped_image_array = image_array[top_boundary:bottom_boundary + 1, :]
        im = Image.fromarray(np.uint8(cropped_image_array))


    width, height = 28, 28
    im = im.resize((width, height))

    if test:
       im = ImageOps.invert(im) # Inversion is needed as the MNIST data set works with black background!

    im = im.convert("L")  # Converting to gray for pixel brightness

    conversion = np.array(im)/255

    return conversion  # returning list of darkness scheme of the picture



def pixelToPicture(image_array):
    """""
    converting array of pixel brightness values, to an actual picture
    """
    width = 28
    height = 28

    # Reshape the 1D array into a 2D array
    pixel_array = np.array(image_array*255)
    image_data = pixel_array.reshape((height, width))

    # Create an image using Pillow
    image = Image.fromarray(image_data.astype('uint8'))

    # Show the image
    image.show()

    # Optionally, save the image to a file
    #image.save('output_image.png')
    return image


def resizeImage(image, test):
    """"
    Normalizing the pictures: fixed white spaces at bottom, top, left and right. Thus centering the number
    """
    vertical_normalized_picture = resizeImageVertical(image, test)

    normalized_picture = resizeImageHorizontal(vertical_normalized_picture, test)

    return normalized_picture


def resizeImageVertical(image, test):
    image_array = parsingPicture(image, test)  # Note that the image has black background, as we started from MNIST

    image_array = image_array.reshape(28, 28)

    non_white_rows = np.where(image_array.max(axis=1) > 0)  # Find the top and bottom boundaries of the non-white area

    top_boundary = non_white_rows[0][0] - 2  # Centering number 2 rows from edge of image
    bottom_boundary = non_white_rows[0][-1] + 2

    if top_boundary < 0:  # If picture is too close to the boundary, we add new null rows to array of image.
        image_array = image_array.tolist()
        null_row = np.array([0 for _ in range(28)])
        for i in range(-top_boundary):
            image_array.insert(0, null_row)
        image_array = np.array(image_array)
        top_boundary = 0  # Null rows are added, thus we need to begin from index zero for the cropping further

    if bottom_boundary - 27 > 0:
        image_array = image_array.tolist()
        null_row = np.array([0 for _ in range(28)])
        for i in range(bottom_boundary - 27):
            image_array.append(null_row)
        image_array = np.array(image_array)
        bottom_boundary = int(10E20)  # cropping at bottom is not needed anymore.


    # Crop the image to the desired boundaries
    cropped_image_array = image_array[top_boundary:bottom_boundary + 1, :]

    if test:
        cropped_image = Image.fromarray(np.uint8(cropped_image_array*255))  # Correct picture formatting
    else:
        cropped_image = Image.fromarray(np.uint8(cropped_image_array))  # Correct picture formatting, no need for 255
                                                                        # as MNIST data set is already normalized

    normalized_image = cropped_image.resize((28, 28))

    return normalized_image


def resizeImageHorizontal(image, test):
    if test:
       image = ImageOps.invert(image)  # We don't have a BLACK background in case of a self taken/drawn picture!

    image_array = parsingPicture(image, test)


    image_array = image_array.reshape(28, 28)

    image_array = np.transpose(image_array)  # Transposing matrix, and applying same logic as in the vertical case

    # Find the top and bottom boundaries of the non-white area
    non_white_rows = np.where(image_array.max(axis=1) > 0)

    top_boundary = non_white_rows[0][0] - 2
    bottom_boundary = non_white_rows[0][-1] + 2


    if top_boundary < 0:  # Adding new lines in case the number is too close to the top
        image_array = image_array.tolist()
        null_row = np.array([0 for _ in range(28)])
        for i in range(-top_boundary):
            image_array.insert(0, null_row)
        image_array = np.array(image_array)
        top_boundary = 0

    if bottom_boundary - 27 > 0: # Adding new lines in case the number is too close to the bottom
        image_array = image_array.tolist()
        null_row = np.array([0 for _ in range(28)])
        for i in range(bottom_boundary - 27):
            image_array.append(null_row)
        image_array = np.array(image_array)
        bottom_boundary = int(10E20)  # Cropping at bottom is not needed anymore

    # Crop the image to the desired boundaries
    cropped_image_array = image_array[top_boundary:bottom_boundary + 1, :]

    cropped_image_array = np.transpose(cropped_image_array)
    if test:
        cropped_image = Image.fromarray(np.uint8(cropped_image_array * 255))  # Correct picture formatting
    else:
        cropped_image = Image.fromarray(np.uint8(cropped_image_array))  # Correct picture formatting

    normalized_image = cropped_image.resize((28, 28))

    return normalized_image


def separatingNumbers(image):
    """""
    Guessing multiple numbers, given a picture of several numbers separated by AT LEAST one white VERTICAL white line
    """""

    splitted_images = []
    #image = ImageOps.invert(image)  # Supposing the background in white, making it compatible with the MNIST data set
    image = image.convert("L")  # Converting to gray for pixel brightness
    image_array = np.array(image)  # Converting image to image array

    image_array = np.transpose(image_array)  # To easier find the white line between the two numbers

    # Find the top and bottom boundaries of the non-white area
    non_white_rows = np.where(image_array.min(axis=1) < 255)  # We have in this case a WHITE background, so we find for
                                                                # lines which at least have a black pixel(value = 255)!
    non_white_rows_indices = non_white_rows[0]

    left_boundary = 0  # Left boundary of the number in question
    for index in range(len(non_white_rows_indices) - 1):
        if non_white_rows_indices[index + 1] - non_white_rows_indices[index] > 10 :  # If white line is found

            split_image_array = image_array[left_boundary: non_white_rows_indices[index] + 1, :]
            split_image_array = np.transpose(split_image_array) # Transposing back


            non_white_rows_splitted_image = np.where(split_image_array.min(axis=1) < 255)
            #non_white_col_splitted_image = np.where(np.transpose(split_image_array).min(axis=1) < 255)

            splitted_image = Image.fromarray(np.uint8(split_image_array))  # Correct picture formatting

            #if not (len(non_white_rows_splitted_image[0]) < 14 or len(non_white_col_splitted_image[0]) < 14):
            if not (len(non_white_rows_splitted_image[0]) < 14):
                splitted_images.append(splitted_image)  # Saving cut image
            left_boundary = non_white_rows_indices[index + 1]  # Re-indexing the top of the picture

    split_image_array = image_array[left_boundary: non_white_rows_indices[-1] + 1, :]

    split_image_array = np.transpose(split_image_array)  # Transposing back

    splitted_image = Image.fromarray(np.uint8(split_image_array))  # Correct picture formatting

    non_white_rows_splitted_image = np.where(split_image_array.min(axis=1) < 255)

    if not (len(non_white_rows_splitted_image[0]) < 14):
        splitted_images.append(splitted_image)


    return splitted_images


def filterBackground(image, test):
    conversion = np.zeros(784)
    threshold = filterBackgroundThreshold(image, 0)
    DIMENSION = image.size
    pix = image.load()
    for row in range(DIMENSION[0]):
        for col in range(DIMENSION[1]):
            if pix[col, row]/255 < threshold and test:  # Filtering out the background

                pix[col, row] = 0
            elif test:
                conversion[row*DIMENSION[1] + col] = pix[col, row]/255
            else:
                conversion[row * DIMENSION[1] + col] = pix[col, row]  # Pixel value already btwn 0-1 in MNIST data set
    return conversion


def filterBackgroundThreshold(image, n_of_sd):
    """"
    We will filter everything which has a brightness
    less than mu + sigma. Above this value will be seen as
    significant.
    """
    image_array = np.array(image)/255  # /255 for normalizing
    data = image_array.reshape((1, image.size[0] * image.size[1]))[0] # Reshaping the np.array and taking the first row
    return np.mean(data) + n_of_sd*np.std(data)

def filterImages(image):
    image = ImageOps.invert(image)
    image = image.convert("L")
    image_array = np.array(image)
    threshold = filterBackgroundThreshold(image, 4.5)
    if threshold > 1:
        threshold = filterBackgroundThreshold(image, 2)

    # Apply the threshold operation using NumPy
    image_array = np.where(image_array < threshold*255, 0, image_array)
    image_array = np.where(image_array >= threshold*255, 255, image_array)

    filtered_image = Image.fromarray(np.uint8(image_array))
    filtered_image = ImageOps.invert(filtered_image)
    return filtered_image

