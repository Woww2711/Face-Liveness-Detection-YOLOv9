import os
import random
import shutil
from itertools import islice

# This file is used when you feel like you have collected enough images for training
# The script below split your data into 3 parts: train, val and test
# It will also create a data.yaml file
# Remember to compress 3 folders and 1 .yaml file to 1 .zip folder

outputFolderPath = "Folder that all images are splited to train,val,test"
inputFolderPath = "Folder that contains all images you want the model to train on"

# Change these as per your need
splitRatio = {"train": 0.8, "val": 0.1, "test": 0.1}
classes = ["replay","real","mask"]

# try:
#     shutil.rmtree(outputFolderPath)
# except OSError as e:
#     os.mkdir(outputFolderPath)

# --------  Directories to Create -----------
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# --------  Get the Names  -----------
listNames = os.listdir(inputFolderPath)

uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))

# --------  Shuffle -----------
random.shuffle(uniqueNames)

# --------  Find the number of images for each folder -----------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# --------  Put remaining images in Training -----------
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

# --------  Split the list -----------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# --------  Copy the files  -----------

sequence = ['train', 'val', 'test']
for i,out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Completed...")


# -------- Creating Data.yaml file  -----------

# REMINDER: BE VERY CAREFUL WHEN CREATING DATA.YAML FILE
# TO TRAIN LOCALLY, YOU NEED TO SPECIFY THE ABSOLUTE PATH
# CHECK THE DATA.YAML PATH THOROUGHLY BEFORE TRAINING, OR IT WILL FAIL AFTER FINAL EPOCH
dataYaml = f'path: ../data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")
