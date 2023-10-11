import random
from PIL import Image, ImageOps


def rotate(im):
    """"
    Rotating image by angle ~ N(0,sigma = 4)
    """
    angle = random.gauss(0, 4)
    return im.rotate(angle)


def displace(im):
    """"
    Displacing image by -1 to 1 image for both x as y direction
    """
    width, height = im.size
    x_translation = random.randint(-1, 1)
    y_translation = random.randint(-2, 2)

    # Create a new image with the same size
    translated_image = Image.new("RGB", (width, height))

    # Paste the original image onto the new image with the desired translation
    translated_image.paste(im, (x_translation, y_translation))
    return translated_image


def noise(im):
    """""
    Changing brightness of pixels of the image according to pixelBrightness ~ N(1, sigma=0.3)
    This causes a noise effect on the picture
    """
    DIMENSION = im.size
    pix = im.load()

    for row in range(DIMENSION[0]):
        for col in range(DIMENSION[1]):
            noise_factor = round(abs(random.gauss(1, 0.3)), 3)
            value_of_pixel = pix[col, row]
            #pix[col, row] = tuple(int(val*noise_factor) for val in value_of_pixel)
            pix[col, row] = int(noise_factor*value_of_pixel)

    return im


# im = Image.open('IMAGES/Test.png')
#
# im = noise(im)
# im.show()
