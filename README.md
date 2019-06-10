# AgeEstimationApp
Training in two ways:  
8 classes classification, predict age interval probability
101 classes classification, predict exact age estimation  

Evaluation:  
accurancy comparision for 8 classes classification   
MAE value calculation for 101 classes classification   


Go to website or run on localhost by running: python server.py & go to http:www.4everyoung.group:5000  

parameters tuning:  
optimizer  
lr rate  
input pixels  
batch size  
depth of layers on top of pretrained model  
svm classifier  
regularizer and regularizer initialization  
data augmentation  
label augmentation  
crop margin  
grey-color  
mixup generator  

.py files:    
utkcamera.py: Real-time Prediction camerer which will be generated in server.py, read in 101 class model: utk.h5  
you can download the file on google drive: https://drive.google.com/drive/u/0/folders/1KfUh3ynBOsggfm9SiDXHKcWFh0cBemNa  
and save them in model folder when you run the server.py to make prediction  
server.py: flask server, read in model of 8 classes: model64.h5  
you can download the file on google drive: https://drive.google.com/drive/u/0/folders/1KfUh3ynBOsggfm9SiDXHKcWFh0cBemNa  
and save them in model folder when you run the server.py to make prediction  
crop.py: perform image detection, cropping and alignment  
preprocess.py: make directories for keras data generator, 101 directories for both training and testing  
               preprocess 8 classes transformation  
model.py: all models we are comparing  
evaluate8classes.py: loading .h5 for accurancy calculation, model ensemble mae calculation and confusion matrix plotting  
plot_history.py: plot mae history for 101 classes classification and 8 classes classification  
mixup_generator.py for 101trainmix.py training processes for 60 epoches  
generator.py for 101trainlabellaug.py training processes for 60 epoches  
distributionCal: calculate Distribution and display plot for age range  
facedetecttest.py: test face detection model with video camera using haarcascade_frontalface_default.xml  
run.sh: automation bash for server deployment on ubuntu machine  

static file including all frontend files  
result file including all confusion matrix and training process plots  

For Random Erasing implementation, we refer to repository :https://github.com/yu4u/cutout-random-erasing  
For mixup generator implementation, we refer to repository :https://github.com/yu4u/mixup-generator  
