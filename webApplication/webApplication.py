from flask import *
from Main import testMIPerCase
import os
from PIL import Image,ImageOps
import base64
import io

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('homePage.html')


@app.route('/upload_canvas', methods = ['POST'])
def upload_canvas():
    """"
    Considering the case where the canvas was used
    """
    try:

        folder_path = 'static/uploads'
        empty_uploads_folder(folder_path)

        data = request.get_json()
        image_data = data.get('image')

        filename = "static/uploads/picture.png" # Saving the image in the corresponding folder
        decoded_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(decoded_data))
        image = image.convert("L")  # Converting the image in the right format
        image.save(filename)
        return redirect('/output')  # Redirecting to the image page

    except Exception as e:
        session['exception'] = str(e)
        return render_template('errorPage.html', error=e)


@app.route('/error', methods=['POST', 'GET'])
def error(**kw):
    return render_template('errorPage.html', error=session['exception'])


@app.route('/output', methods=['POST', 'GET'])
def output():
    if request.method == "GET":


        filename = 'static/uploads/picture.png'

        folder_path = 'static/plots'
        empty_uploads_folder(folder_path)

        folder_path = 'static/normalizedPictures'
        empty_uploads_folder(folder_path)

        guess, certainty = testMIPerCase(filename)
        print(guess, "CERTAINTY:", certainty)
        return render_template('output.html', guess=guess, certainty=certainty)

    else:
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                file_extension = os.path.splitext(image.filename)[-1].lower()
                print(file_extension)

                filename = 'static/uploads/picture.png'
                folder_path = 'static/uploads'
                empty_uploads_folder(folder_path)
                image.save(filename)

                if file_extension in ['.jpg']:
                    img = Image.open(filename)
                    img = img.rotate(-90, expand=True)  # Getting rid of orientation metadata
                    empty_uploads_folder(folder_path)
                    img.save(filename)

                folder_path = 'static/plots'
                empty_uploads_folder(folder_path)

                folder_path = 'static/normalizedPictures'
                empty_uploads_folder(folder_path)

                guess, certainty = testMIPerCase(filename)

                return render_template('output.html', guess=guess, certainty=certainty)


    return 'No image selected or uploaded.'


def empty_uploads_folder(folder_path):
    # Empty the 'uploads' folder by deleting its contents
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print("[DELETED]")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    return

if __name__ == '__main__':
    app.run(debug=False, host='localhost')
