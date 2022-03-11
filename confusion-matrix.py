import numpy as np
import pandas as pd
import dataframe_image as dfi
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model
import os

load_path_covid = "E:/my_projects/covid-project/Project-code/source/test/covid"
load_path_pneumonia = "E:/my_projects/covid-project/Project-code/source/test/pneumonia"
load_path_normal = "E:/my_projects/covid-project/Project-code/source/test/normal"

model = load_model('models/model-054-0.960769-0.892058.h5') # Loding the trained model
img_width, img_height = 224, 224  # Default input size for VGG16
conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(img_width, img_height, 3))


def predict(img_path):

    org_img = image.load_img(img_path)
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
        
    classes = ["Covid19", "Normal", "Pneumonia"]
    result = str(classes[np.argmax(np.array(prediction[0]))])

    return result

#Confusion matrix values for covid

img = os.listdir(load_path_covid)

covid,normal,pneumonia = 0,0,0
print("Testing Covid images")
for i in range(len(img)):
    result = predict(load_path_covid+"/"+img[i])
    if result == "Covid19":
        covid += 1
    elif result == "Normal":
        normal += 1
    else:
        pneumonia += 1
covid_result = {"covid":covid,"normal":normal,"pneumonia":pneumonia}
cov_row = [covid,normal,pneumonia]
print(covid_result)

#Confusion matrix values for normal

img = os.listdir(load_path_normal)

covid,normal,pneumonia = 0,0,0
print("Testing normal images")
for i in range(len(img)):
    result = predict(load_path_normal+"/"+img[i])
    if result == "Normal":
        normal += 1
    elif result == "Pneumonia":
        pneumonia += 1
    else:
        covid += 1
normal_result = {"covid":covid,"normal":normal,"pneumonia":pneumonia}
norm_row = [covid,normal,pneumonia]
print(normal_result)

#Confusion matrix values for pneumonia

img = os.listdir(load_path_pneumonia)

covid,normal,pneumonia = 0,0,0
print("Testing pneumonia images")
for i in range(len(img)):
    result = predict(load_path_pneumonia+"/"+img[i])
    if result == "Pneumonia":
        pneumonia += 1
    elif result == "Normal":
        normal += 1
    else:
        covid += 1
pneumonia_result = {"covid":covid,"normal":normal,"pneumonia":pneumonia}
pneu_row = [covid,normal,pneumonia]
print(pneumonia_result)

print("\nConfusion Matrix")

df = pd.DataFrame(columns=('Covid','Normal','Pneumonia'))
df.loc['Covid'] = cov_row
df.loc['Normal'] = norm_row
df.loc['Pneumonia'] = pneu_row

dfs = df.style.background_gradient()
dfi.export(dfs,'confusion-matrix-styled.png')
dfi.export(df,'confusion-matrix.png')

print(df)
print(dfs)