import cv2
import numpy as np
import io
import base64
from PIL import Image
from flask import Flask, flash, request, redirect, render_template
from function.fruits_counter import counter

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = 'uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('File tidak ada')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang terupload')
        return redirect(request.url)
    if file and allowed_file(file.filename):

        filestr = request.files['file'].read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

        img, num = counter(image)
        flash("jumlah buah: "+str(num))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        file_object = io.BytesIO()
        img= Image.fromarray(img)
        img.save(file_object, 'PNG')
        base64img = "data:image/png;base64,"+base64.b64encode(file_object.getvalue()).decode('ascii')

        return render_template('upload.html', image=base64img )
    else:
        flash('Hanya dapat memproses file -> png, jpg, jpeg')
        return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)