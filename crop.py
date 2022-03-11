from PIL import Image as img
import os

#CROP TRAIN
path = 'E:/my_projects/covid-project/Project-code/downsample/train'
_image = os.listdir(path)

a,b,c=0,0,0
print("No of images present : ",len(_image))

left, top, right, bottom = 40, 100, 460, 500
for i in range(len(_image)):
    print("crop", i+1)
    image = img.open(path+"/"+_image[i])
    c_image = image.crop((left, top, right, bottom))
    if _image[i][:3] == "nor":
        c_image.save("source/train/normal/crop_normal"+str(a+1)+".png")
        a += 1
        print("cn_Image ",a," saved")
    elif _image[i][:3] == "cov":
        c_image.save("source/train/covid/crop_covid"+str(b+1)+".png")
        b += 1
        print("cc_Image ",b," saved")
    else:
        c_image.save("source/train/pneumonia/crop_pneumonia"+str(c+1)+".png")
        c += 1
        print("cp_Image ",c," saved")

path = 'E:/my_projects/covid-project/Project-code/downsample/test'
_image = os.listdir(path)

#CROP TEST
a, b, c = 0, 0, 0
print("No of images present : ", len(_image))

left, top, right, bottom = 40, 100, 460, 500
for i in range(len(_image)):
    print("crop", i+1)
    image = img.open(path+"/"+_image[i])
    c_image = image.crop((left, top, right, bottom))
    if _image[i][:3] == "nor":
        c_image.save("source/test/normal/crop_normal"+str(a+1)+".png")
        a += 1
        print("cn_Image ", a, " saved")
    elif _image[i][:3] == "cov":
        c_image.save("source/test/covid/crop_covid"+str(b+1)+".png")
        b += 1
        print("cc_Image ", b, " saved")
    else:
        c_image.save("source/test/pneumonia/crop_pneumonia"+str(c+1)+".png")
        c += 1
        print("cp_Image ", c, " saved")

