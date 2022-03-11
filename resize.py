from PIL import Image as img
import os

norm_train_path = 'E:/my_projects/covid-project/Project-code/train/NORMAL'
normal_train_img = os.listdir(norm_train_path)

pneu_train_path = 'E:/my_projects/covid-project/Project-code/train/PNEUMONIA'
pneu_train_img = os.listdir(pneu_train_path)

norm_test_path = 'E:/my_projects/covid-project/Project-code/test/NORMAL'
normal_test_img = os.listdir(norm_test_path)

pneu_test_path = 'E:/my_projects/covid-project/Project-code/test/PNEUMONIA'
pneu_test_img = os.listdir(pneu_test_path)

covid_path = 'E:/my_projects/covid-project/Project-code/covid-images'
covid_img = os.listdir(covid_path)

#Defining NEW Size to resize
new_size = (500, 500)

#Resize Normal train Images
for i in range(len(normal_train_img)):
    print("resize - normal",i+1)
    image = img.open(norm_train_path+"/"+normal_train_img[i])
    r_image = image.resize(new_size, img.BICUBIC)
    r_image.save("downsample/train/normal"+str(i)+".png")
    print("resize - normal train saved", i+1)

#Resize Normal test Images
for i in range(len(normal_test_img)):
    print("resize - normal", i+1)
    image = img.open(norm_test_path+"/"+normal_test_img[i])
    r_image = image.resize(new_size, img.BICUBIC)
    r_image.save("downsample/test/normal"+str(i)+".png")
    print("resize - normal test saved", i+1)
    
#Resize Covid Images
for i in range(len(covid_img)):
    print("resize - covid", i+1)
    image = img.open(covid_path+"/"+covid_img[i])
    r_image = image.resize(new_size, img.BICUBIC)
    if i%3 == 0:
        r_image.save("downsample/test/covid"+str(i)+".png")
        print("resize - covid test saved", i+1)
    else:
        r_image.save("downsample/train/covid"+str(i)+".png")
        print("resize - covid train saved", i+1)

#Resize Pneumonia train Images
for i in range(len(pneu_train_img)):
    print("resize - pneumonia", i+1)
    image = img.open(pneu_train_path+"/"+pneu_train_img[i])
    r_image = image.resize(new_size, img.BICUBIC)
    r_image.save("downsample/train/pneumonia"+str(i)+".png")
    print("resize - pneumonia train saved", i+1)

#Resize Pneumonia test Images
for i in range(len(pneu_test_img)):
    print("resize - pneumonia", i+1)
    image = img.open(pneu_test_path+"/"+pneu_test_img[i])
    r_image = image.resize(new_size, img.BICUBIC)
    r_image.save("downsample/test/pneumonia"+str(i)+".png")
    print("resize - pneumonia test saved", i+1)


