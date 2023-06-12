from flask import Flask,request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from utils import predict_util

app = Flask(__name__)

# @app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('os.path.dirname(os.path.abspath(__file__))'+'image.jpg')
        noisy_img = predict_util('os.path.dirname(os.path.abspath(__file__))'+'image.jpg')
    if request.method == 'GET':
        request.GET('os.path.dirname(os.path.abspath(__file__))'+'noisy.jpg')
    return 'success'

# predict_util()

if __name__ == 'main': 
    app.run()
    