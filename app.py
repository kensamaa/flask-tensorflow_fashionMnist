from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(filename):
    # Load the image
    img = keras.preprocessing.image.load_img(filename, grayscale=True, target_size=(28, 28))
    # Convert the image to array
    img = keras.preprocessing.image.img_to_array(img)
    # Reshape the image 
    img = img.reshape(1, 28, 28, 1)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')
@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)
                img = read_image(file_path)
                # Predict the class of an image
                with graph.as_default():
                  model1 = keras.models.load_model('clothing_classification_model.h5')
                  class_prediction = model1.predict_classes(img)
                print(class_prediction)
                if class_prediction[0] == 0:
                  product = "T-shirt/top"
                elif class_prediction[0] == 1:
                  product = "Trouser"
                elif class_prediction[0] == 2:
                  product = "Pullover"
                elif class_prediction[0] == 3:
                  product = "Dress"
                elif class_prediction[0] == 4:
                  product = "Coat"
                elif class_prediction[0] == 5:
                  product = "Sandal"
                elif class_prediction[0] == 6:
                  product = "Shirt"
                elif class_prediction[0] == 7:
                  product = "Sneaker"
                elif class_prediction[0] == 8:
                  product = "Bag"
                else:
                  product = "Ankle boot"
                return render_template('predict.html', product = product, user_image = file_path)
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."
    return render_template('predict.html')

if __name__ == "__main__":
    global graph
    graph = tf.compat.v1.get_default_graph()
    app.run()