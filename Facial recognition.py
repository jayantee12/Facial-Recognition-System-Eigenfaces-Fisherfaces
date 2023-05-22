from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image
import math
import colorama
from colorama import Fore

#DISPLAYING SOME REFERENCE IMAGES FROM DATASET
plt.figure()
fig, ax = plt.subplots(4,10, figsize=(100,50), sharex=True)

for j in [5,6,7,8]:
    for i in range(1,11):
        f = "D:/SEM 2/Project2/s"+str(j)+"/"+str(i)+".pgm"
        im= Image.open(f)
        ax[j-5,i-1].imshow(im, cmap='gray')
        
#FUNCTION TO DISPLAY IMAGES IN THE OUTPUT FOR COMPARISON LATER

def DISPLAY(path1,path2):
    plt.figure()
    fig, ax = plt.subplots(1,2, figsize=(5,2.5), sharex=True)
    fig.suptitle('Wrong classification:')
    im1= Image.open(path1)
    im2=Image.open(path2)
    ax[0].imshow(im1, cmap='gray')
    ax[1].imshow(im2, cmap='gray')
 


#FUNCTION TO COLLECT IMAGE OBJECTS AND THEIR LABELS FROM DIRECTORY AS NUMPY ARRAYS FOR LATER USE
def get_images(path):
    paths = [os.path.join(path, l) for l in os.listdir(path)]
    images=[]
    labels=[]
    x=[]
    for i in paths:
        a=os.path.split(i)[1]
        for j in range(10):
            im_path=i + '/' + str(j+1) +'.pgm'
            image = Image.open(im_path)
            image = np.array(image, 'uint8')
            image=image.flatten()
            images.append(image)
            x.append(j+1)
            labels.append(int(str(a)[1:]))
    images = np.array(images, 'uint8')
    labels = np.array(labels)
    x = np.array(x)
    return images,labels,x

  
  
  # SPLITTING ALL OF THE DATA INTO 80% TRAINING DATA - 20% TEST DATA

def split_training_and_test(images,labels,x):

    images_train, images_test, labels_train, labels_test, index_train, index_test = train_test_split(images, labels, x, 0.2, random_state = 0)
    
    return images_train, images_test, labels_train, labels_test, index_train, index_test
 


  #EIGENFACE MODEL: CALCULATING OPTIMAL W (EIGENVECTORS)

def Eigenface_model(images_train, labels_train):
    
    #subtracting mean face from all faces
    
    org=images_train
    images_train = images_train.T
    mean = images_train.mean(axis=1, keepdims=True)
    images_train = images_train - mean
    
    #computing covariance matrix (Total Scatter matrix)
    
    cov = np.matmul(np.transpose(images_train), images_train)
    
    #getting the eigen values and eigen vectors for Scatter matrix
    
    eigval, eigv = np.linalg.eig(cov)
    
    #getting corresponding eigen vectors
    eigu = np.matmul(images_train, eigv.real)
    
    #normalizing eigen vectors by dividing by root(Eigenvalue*Number of images)
    
    a=np.reciprocal(eigval)
    a=np.sqrt(a)
    eigu=np.multiply(eigu,a)
    eigu=1/math.sqrt(len(images_train))*eigu
    

    #arranging eigen values and hence eigen vecs in descending order of eigen values TO GET m LARGEST VALUES LATER
    
    idx = np.argsort(-eigval)
    eigval = eigval[idx]
    eigu = eigu[:, idx]
    

    return eigval, eigu, mean, org
  
  
  
  #APPLYING THE ALGORITHM ON TEST IMAGES AND DISPLAYING RESULTS

def Eigenface_test(images_test, labels_test, labels_train, index_train, index_test, weights, mbest, mean):

    #Normalizing the test images
    
    images_test = images_test.T- mean
    labels_test = labels_test.T

    #calculating projected test images
    
    testweights = np.matmul(mbest.T, images_test)
    correct=0
    
    for i in range(0, len(labels_test)):
        
        #calculating sum of square of error error for each test image (FOR DISTANCE FROM CLASSES)
        
        testweight = np.resize(testweights[:, i], (testweights.shape[0], 1))
        err = (weights - testweight) ** 2
        ssq1 = np.sum(err ** (1/2), axis=0)
        

        #Finding the closest face to the test image (MINIMUM SSQ)
        
        dist= ssq1.min(axis=0, keepdims=True)
        match=labels_train[ssq1.argmin(axis=0)]

        #Checking if results are correct and Printing the result for this image
        t = "D:/SEM 2/Project2/s"+str(labels_test[i])+"/"+str(index_test[i])+".pgm"
        m= "D:/SEM 2/Project2/s"+str(match)+"/"+str(index_train[i])+".pgm"
        
        if dist < 250:
            if labels_test[i] == match:
                correct+=1
                print("subject %s identified correctly as %s with distance %f" 
                      %(labels_test[i], match, dist.real))

            else:
                
                print ("subject %s identified incorrectly as %s with distance %f"
                       %(labels_test[i], match, dist.real))
                
                DISPLAY(t,m)
        else:
            print ("subject face not match in database :")
            display(Image.open(t))
            
    print(Fore.RED+ "\033[1m"+ "\n THE FINAL ACCURACY OF EIGENFACES IS %f PERCENT" 
          % (correct*100 / len(labels_test))+ "\033[0;0m")
    return
  
  
#MAIN PROGRAM: EIGENFACES (Calling all functions)

#Getting the images and labels from Directory

path=('D:/SEM 2/Project2/')
images, labels, x= get_images(path)

#Splitting the data into training and test

images_train, images_test, labels_train, labels_test, index_train, index_test = split_training_and_test(images,labels,x)


#Performing Eigenface analysis and get Eigenface vectors

eigval, eigu, mean, org= Eigenface_model(images_train, labels_train)


#Finding m best eigen vectors for dimension reduction to m dimensions

sum1 = np.sum(eigval, axis=0)
k = 0
for m in range(0, len(labels_train)):
    k += eigval[m] / sum1
    if k > 0.95:
        break
kbest = eigu[:, 0:m]

#Getting the projections of the of eigenfaces for each input image

weights = np.matmul(kbest.T, images_train.T- mean)


#Testing the Eigenface model on the test images and print the result

Eigenface_test(images_test, labels_test, labels_train, index_train, index_test, weights, kbest, mean)









#FISHERFACE MODEL: FINDING OUT BOTH SCATTER MATRICES AND DOING EIGEN ANALYSIS TO TAKE C-1 EIGENVALUES, 
#EIGENVECTORS FOR DIMENSION REDUCTION

def Fishers_model(images, labels):
    
    #Getting the shape of image variable and the no of classes
    
    d = images.shape [1]
    classes = np.unique(labels)
    
    
    #Initailising the two covariance matrix, Sw- within cluster, Sw- Between clusters
    
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    
    #Getting the mean of all images 
    totmean=images.mean(axis=0, keepdims=True)
    
    
    #Calculating and updating Sw and Sb according to their formulas
    
    for i in range(0, len(classes)):
        imagesi = images[np.where(labels == i+1)[0], :]
        ni = imagesi.shape[0]
        MEANi = imagesi.mean(axis=0, keepdims=True)
        Sw = Sw + np.matmul((imagesi - MEANi).T, (imagesi - MEANi))
        Sb = Sb + ni * np.matmul((MEANi - totmean).T, (MEANi - totmean))
        
        
    #Getting the generalized eigen values and eigen vectors of the matrix Inv(Sw)*Sb
    
    eigval_fld, eigvec_fld = np.linalg.eig(np.linalg.inv(Sw) * Sb)
    
    
    #Sorting eigen vectors and eigen values in decreasing order of the eigenvalues
    idx = np.argsort(-eigval_fld)
    eigval_fld = eigval_fld[idx]
    eigvec_fld = eigvec_fld[:, idx]
    
    #Taking the first c-1 eigenvalues and eigenvectors for dimension reduction to c-1
    
    eigval_fld = eigval_fld[0:(len(classes)-1)]
    eigvec_fld = eigvec_fld[:,0:(len(classes)-1)]
    
    
    return eigval_fld, eigvec_fld
  
  
  #TESTING THE FISHERS MODEL ON TEST IMAGES AND DISPLAYING RESULTS

def Fishers_test(images_train, images_test, labels_test, labels_train, index_train, index_test, eigvec, mean):
    
    #Getting the projected training images
    
    weights = np.matmul(eigvec.T, images_train)
    
    #Normalizing the test images
    
    images_test = images_test.T - mean
    labels_test = labels_test.T

    #calculating the projected test images
    
    testweights = np.matmul(eigvec.T, images_test)

    correct = 0
    
    #Calculating the results for each test image 
    
    for i in range(0, len(labels_test)):
        
        #calculating error for each test image
        testweight = np.resize(testweights[:, i], (testweights.shape[0], 1))
        err = (weights - testweight) ** 2

        #calculating the sum of square of error
        ssq1 = np.sum(err **(1/2), axis=0)

        #Finding the closest face to the test image
        
        dist = ssq1.min(axis=0, keepdims=True)
        match = labels_train[ssq1.argmin(axis=0)]

        #Checking if results are correct and printing the results for this image
        
        t = "D:/SEM 2/Project2/s"+str(labels_test[i])+"/"+str(index_test[i])+".pgm"
        m= "D:/SEM 2/Project2/s"+str(match)+"/"+str(index_train[i])+".pgm"
        
        if dist < 230:
            if labels_test[i] == match:
                correct+=1
                print("subject %d identified correctly as %d with distance %f" 
                      % (labels_test[i], match, dist.real))
            else:
                print("subject %d identified incorrectly as %d with distance %f" 
                      % (labels_test[i], match, dist.real))
                DISPLAY(t,m)
        else:
            print("subject face not match in database:")
            display(Image.open(t))
            
    print(Fore.RED+"\033[1m"+"\n The accuracy of Fisherfaces is %f percent" 
          %(correct*100/len(labels_test))+"\033[0;0m")
    
    return weights
  
  
  #MAIN PROGRAM: FISHERFACES (Calling all functions)

#Getting the images and labels from directory

path=('D:/SEM 2/Project2/')
images, labels,x= get_images(path)


#Splitting the data into training and test

images_train, images_test, labels_train, labels_test, index_train, index_test = split_training_and_test(images,labels,x)

#Applying Eigenface model first to reduce dimensions to N-c

eigval_pca, eigvec_pca, mean, org= Eigenface_model(images_train, labels_train)
[n,d]= images_train.shape
c = len(np.unique(labels_train))

eigvec_pca1 = eigvec_pca[:,0: n-c]

#Projecting the images onto N-c dimension using PCA
images_train = images_train.T
images_train = images_train - mean
images_train_project = np.matmul(images_train.T, eigvec_pca1)



#Projecting to c-1 dimension using FLD on the already PCA-projected images of N-c dimensions

eigval_fld, eigvec_fld = Fishers_model( images_train_project, labels_train)


#Getting the total final eigen vectors by multiplying the eigen vectors from PCA and FLD
eigvec= np.matmul(eigvec_pca1,eigvec_fld)


#Testing the Fisherface model on the test images and displaying the result
weights=Fishers_test(images_train, images_test, labels_test, labels_train, index_train, index_test, eigvec, mean)





