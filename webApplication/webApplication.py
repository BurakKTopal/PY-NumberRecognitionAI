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
    try:
        folder_path = 'static/uploads'
        empty_uploads_folder(folder_path)

        data = request.get_json()
        image_data = data.get('image')
        # Process the image_data as needed (e.g., save it to a file or a database)
        # For simplicity, we'll save it to a file
        filename = "static/uploads/picture.png"
        decoded_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(decoded_data))
        print(image)
        image = image.convert("L")
        image.save(filename)


        return redirect('/output')

    except Exception as e:
        return 'Error uploading image: ' + str(e)




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
                # Save the uploaded image to a folder (e.g., "uploads")
                #filename = 'static/uploads/' + image.filename

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
    app.run(debug=True, host='192.168.1.41')
    #app.run(debug=True, host='10.5.11.176')
    #app.run(debug=True, host='192.168.116.57')
