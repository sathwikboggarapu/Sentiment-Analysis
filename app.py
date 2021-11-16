import os
from flask import Flask, app, request
from flask.templating import render_template
from werkzeug.utils import secure_filename

# from .scripts import analysis

ANALYSIS_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ANALYSIS_FOLDER

@app.route('/')
def homepage():
    return render_template('landing_page.html')

@app.route('/input', methods=["POST"])
def input_images():
    if request.method == "POST":
        f = request.files['file']
        f.save("static\\files\\tmp.csv")
#        analysis.execute_analysis()
        return render_template('inputs.html')
        

@app.route('/outputs')
def output_images():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], 'image1.png')
    filepath3 = os.path.join(app.config['UPLOAD_FOLDER'], 'image2.png')


    return render_template('outputs.html', image = filepath, image1 = filepath2, image2 = filepath3)