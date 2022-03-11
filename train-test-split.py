from PIL import Image as img
import os

#NORMAL TRAIN
path = 'E:/my_projects/covid-project/Project-code/train/NORMAL'

imge = os.listdir(path)

for i in range(len(imge)):
    print("normal ", i+1)
    image = img.open(path+"/"+imge[i])
    image.save("source/train/normal/normal"+str(i+1)+".png")

#PNEUMONIA TRAIN
path = 'E:/my_projects/covid-project/Project-code/train/PNEUMONIA'

imge = os.listdir(path)

for i in range(len(imge)):
    print("pneumonia ", i+1)
    image = img.open(path+"/"+imge[i])
    image.save("source/train/pneumonia/pneumonia"+str(i+1)+".png")

#COVID-19 TRAIN
path = 'E:/my_projects/covid-project/Project-code/covid-images'

imge = os.listdir(path)

for i in range(750):
    print("covid ", i+1)
    image = img.open(path+"/"+imge[i])
    image.save("source/train/covid/covid"+str(i+1)+".png")

#COVID-19 TEST
path = 'E:/my_projects/covid-project/Project-code/covid-images'

imge = os.listdir(path)

for i in range(700, 930):
    print("covid ", i+1)
    image = img.open(path+"/"+imge[i])
    image.save("source/test/covid/covid"+str(i+1)+".png")

#NORMAL TEST
path = 'E:/my_projects/covid-project/Project-code/test/NORMAL'

imge = os.listdir(path)

for i in range(len(imge)):
    print("normal ", i+1)
    image = img.open(path+"/"+imge[i])
    image.save("source/test/normal/normal"+str(i+1)+".png")

#PNEUMONIA TEST
path = 'E:/my_projects/covid-project/Project-code/test/PNEUMONIA'

imge = os.listdir(path)

for i in range(len(imge)):
    print("pneumonia ", i+1)
    image = img.open(path+"/"+imge[i])
    image.save("source/test/pneumonia/pneumonia"+str(i+1)+".png")
