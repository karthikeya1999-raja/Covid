from flask import Flask, render_template,request
import numpy as np
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model


app = Flask(__name__)

model = load_model('models/model-054-0.960769-0.892058.h5')
img_width, img_height = 224, 224  # Default input size for VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))

def max(a):
    maxi = 0
    for ele in a:
        if ele>maxi:
            maxi = ele
    return maxi

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        name= request.files['file']
        print(name.filename)
        img_path = 'uploads/'+name.filename
        name.save(img_path)
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
        img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application

        # Extract features
        features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))

        # Make prediction
        try:
            prediction = model.predict(features)
        except:
            prediction = model.predict(features.reshape(1, 7*7*512))

        print(max(prediction[0]))
        
        classes = ["Covid19", "Normal", "Pneumonia"]
        pred_class = str(classes[np.argmax(np.array(prediction[0]))])

        org_img = image.load_img(img_path)
        plt.imshow(org_img)                           
        plt.axis('off')
        plt.title(pred_class)
        plt.show()

    return {"prediction":pred_class,"accuracy":str(max(prediction[0]))}

app.run(debug=True)