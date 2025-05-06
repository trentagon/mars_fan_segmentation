
import os
import sys
import glob
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tifffile 
from math import floor, ceil, sqrt, exp
import time
import argparse
from networksVaryingKernel import Model4ChannelInitialToMiddleLayersDifferent,Model5ChannelInitialToMiddleLayersDifferent,Model6ChannelInitialToMiddleLayersDifferent
from skimage.transform import resize
from skimage import filters
from skimage import morphology
from skimage.filters import rank
from skimage.transform import resize
from skimage import filters
from skimage.morphology import disk
from tifffile import imsave
from sklearn.metrics import confusion_matrix
import warnings 
warnings.filterwarnings('ignore')

def train_model(inputImage, trainingEpochs=2, batchSize=8, finalFeatures=8, patchSize=224):
    
    data = np.copy(inputImage)
    
    # Set manual seed 
    manualSeed=85
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)

    # Constant parameters
    modelInputMean = 0
    trainingBatchSize = batchSize
    maxIter = 50  ##number of maximum iterations over same batch
    lr = 0.001 ##Learning rate           

    # User parameters
    nFeaturesIntermediateLayers = 64  ##number of features in the intermediate layers
    nFeaturesFinalLayer = finalFeatures ##number of features of final classification layer
    numTrainingEpochs = trainingEpochs
    modelName = 'Model5ChannelInitialToMiddleLayersDifferent'
    numberOfImageChannels = data.shape[2]
    trainingPatchSize = patchSize ##Patch size used for training self-sup learning
    trainingStrideSize = int(trainingPatchSize)

    # Parallelize code if avaialable
    useCuda = torch.cuda.is_available()
    
    class TrainingDatasetLoader(torch.utils.data.Dataset):
    ##loads data for self-supervised model learning
        def __init__(self, data, useCuda, patchSize = 112, stride = None, transform=None):
              #Initialization
              self.transform = transform

              ##Torchvision data transforms
              self.GaussianBlur = transforms.GaussianBlur(5, sigma=(0.1, 2.0))

              # basics
              self.transform = transform
              self.patchSize = patchSize
              if not stride:
                self.stride = 1
              else:
                self.stride = stride

              ##Converting from Row*Col*Channel format to Channle*Row*Col
              data = np.transpose(data, (2, 0, 1))

              self.data = torch.from_numpy(data)
              self.useCuda = useCuda
              self.data = self.data.type(torch.FloatTensor)


              # calculate the number of patches
              s = self.data.shape
              n1 = ceil((s[1] - self.patchSize + 1) / self.stride)
              n2 = ceil((s[2] - self.patchSize + 1) / self.stride)
              n_patches_i = n1 * n2
              self.n_patches = n_patches_i

              self.patch_coords = []          
              # generate path coordinates
              for i in range(n1):
                    for j in range(n2):
                        # coordinates in (x1, x2, y1, y2)
                        current_patch_coords = ( 
                                        [self.stride*i, self.stride*i + self.patchSize, self.stride*j, self.stride*j + self.patchSize],
                                        [self.stride*(i + 1), self.stride*(j + 1)])
                        self.patch_coords.append(current_patch_coords)

        def __len__(self):
              #Denotes the total number of samples of training dataset
              return self.n_patches

        def __getitem__(self, idx):
              current_patch_coords = self.patch_coords[idx]
              limits = current_patch_coords[0]

              I1 = self.data[:, limits[0]:limits[1], limits[2]:limits[3]]
              randomTransformation = torch.randint(low=0,high=2,size=(1,)) ##here high is one above the highest integer to be drawn from the distribution.
              if randomTransformation == 0:
                 I2 = self.GaussianBlur(I1)
              elif randomTransformation == 1:
                 I2 = I1.clone()
                 I2[1,:,:]=0
              sample = {'I1': I1,'I2': I2}

              if self.transform:
                sample = self.transform(sample)

              return sample
        
    # MAIN MODEL TRAINING CODE------
    if modelName=='Model4ChannelInitialToMiddleLayersDifferent':
        model = Model4ChannelInitialToMiddleLayersDifferent(numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer) 
    elif modelName=='Model5ChannelInitialToMiddleLayersDifferent':
        model = Model5ChannelInitialToMiddleLayersDifferent(numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer)
    elif modelName=='Model6ChannelInitialToMiddleLayersDifferent':
        model = Model6ChannelInitialToMiddleLayersDifferent(numberOfImageChannels,nFeaturesIntermediateLayers,nFeaturesFinalLayer)
    else:
        sys.exit('Unrecognized model name')
    #print(model)

    if useCuda:
        model.cuda()

    model.train()
    
    lossFunction = torch.nn.CrossEntropyLoss()
    lossFunctionSecondary = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) ##Adam or SGD
    #optimizer = optim.Adam(model.parameters(), lr=lr)  ##Adam or SGD

    trainingDataset = TrainingDatasetLoader(data, useCuda, patchSize = trainingPatchSize, stride = trainingStrideSize, transform=None)
    trainLoader = torch.utils.data.DataLoader(dataset=trainingDataset,batch_size=trainingBatchSize,shuffle=True) 

    # lossPrimary1Array = torch.empty((1))
    # lossPrimary2Array = torch.empty((1))
    # lossPrimaryArray = torch.empty((1))
    # lossSecondary1Array = torch.empty((1))
    # lossSecondary2Array = torch.empty((1))
    # lossTotalArray = torch.empty((1))

    for epochIter in range(numTrainingEpochs):
        for batchStep, batchData in enumerate(trainLoader):

            if useCuda:
                data1ForModelTraining = batchData['I1'].float().cuda()
                data2ForModelTraining = batchData['I2'].float().cuda()
            else:
                data1ForModelTraining = batchData['I1'].float()
                data2ForModelTraining = batchData['I2'].float()
            
            randomShufflingIndices = torch.randperm(data2ForModelTraining.shape[0])
            data2ForModelTrainingShuffled = data2ForModelTraining[randomShufflingIndices,:,:,:]

            for trainingInsideIter in range(maxIter):
                optimizer.zero_grad()

                projection1, projection2 = model(data1ForModelTraining, data2ForModelTraining)
                _,projection2Shuffled = model(data1ForModelTraining,data2ForModelTrainingShuffled)

                _,prediction1 = torch.max(projection1,1)
                _,prediction2 = torch.max(projection2,1)
                _,prediction2Shuffled = torch.max(projection2Shuffled,1)

                lossPrimary1 = lossFunction(projection1, prediction1) 
                lossPrimary2 = lossFunction(projection2, prediction2)

                lossSecondary1 = lossFunctionSecondary(projection1,projection2)
                lossSecondary2 = -lossFunctionSecondary(projection1,projection2Shuffled)

                lossPrimary = (lossPrimary1+lossPrimary2)/2
                lossTotal = (lossPrimary1+lossPrimary2+lossSecondary1+lossSecondary2)/4
                #lossTotal = (lossPrimary1+lossPrimary2+lossSecondary1)/3

                # lossPrimary1Array = torch.cat((lossPrimary1Array,lossPrimary1.unsqueeze(0).cpu().detach()))
                # lossPrimary2Array = torch.cat((lossPrimary2Array,lossPrimary2.unsqueeze(0).cpu().detach()))
                # lossPrimaryArray = torch.cat((lossPrimaryArray,lossPrimary.unsqueeze(0).cpu().detach()))
                # lossSecondary1Array = torch.cat((lossSecondary1Array,lossSecondary1.unsqueeze(0).cpu().detach()))
                # lossSecondary2Array = torch.cat((lossSecondary2Array,lossSecondary2.unsqueeze(0).cpu().detach()))
                # lossTotalArray = torch.cat((lossTotalArray,lossTotal.unsqueeze(0).cpu().detach()))

                if epochIter==0:
                    lossPrimary.backward()
                else:
                    lossTotal.backward()

                optimizer.step()
            #print ('Epoch: ',epochIter, '/', numTrainingEpochs, 'batch: ',batchStep)
        #print('End of epoch', epochIter)
    #torch.save(model,saveModelPath)
    
    ###There is an extra zero in loss arrays at beginning
    # lossPrimary1Array = lossPrimary1Array[1:-1] 
    # lossPrimary2Array = lossPrimary2Array[1:-1] 
    # lossPrimayArray = lossPrimaryArray[1:-1]
    # lossSecondary1Array = lossSecondary1Array[1:-1] 
    # lossSecondary2Array = lossSecondary2Array[1:-1] 
    # lossTotalArray = lossTotalArray[1:-1]
    
    return model

def segment_image(inputImage, inputModel):
    
    eps = 1e-14
    noclutter=False  ##ignore clutter in computing accuracy
    palette = {0 : (255, 255, 255), # Impervious surfaces (white)
               1 : (0, 0, 255),     # Buildings (blue)
               2 : (0, 255, 255),   # Low vegetation (cyan)
               3 : (0, 255, 0),     # Trees (green)
               4 : (255, 255, 0),   # Cars (yellow)
               5 : (255, 0, 0),     # Clutter/background (red)
               6 : (255, 0, 255),   # (pink)
               7 : (0, 0, 0)}       # Undefined (black)

    invert_palette = {v: k for k, v in palette.items()}
    
    model = inputModel  
    
    useCuda = torch.cuda.is_available()
    if useCuda:
          model.cuda()

    ##Shape of input image
    inputImageShape = inputImage.shape
    inputImageShapeRow = inputImageShape[0]
    inputImageShapeCol = inputImageShape[1]

    segmentationMap1 = np.zeros((inputImageShapeRow,inputImageShapeCol))

    segmentationStride = 800  ##since cannot process the entire image at a time
    segmentationOverlap = 100  ## intentional overlap, since cannot process the entire image at a time
    for imageRowIter in range(0,inputImageShapeRow,segmentationStride):  ##since Potsdam images are of size 6000*6000
        for imageColIter in range(0,inputImageShapeCol,segmentationStride):  ##since Potsdam images are of size 6000*6000

            if imageRowIter==0 and imageColIter==0:
                startingRowIndex = 0
                endingRowIndex = segmentationStride+segmentationOverlap
                startingColIndex = 0
                endingColIndex = segmentationStride+segmentationOverlap
            elif imageRowIter==0:
                startingRowIndex = 0
                endingRowIndex = segmentationStride+segmentationOverlap
                startingColIndex = imageColIter-segmentationOverlap
                endingColIndex = min(imageColIter+segmentationStride+segmentationOverlap,inputImageShapeCol)
            elif imageColIter==0:
                startingRowIndex = imageRowIter-segmentationOverlap
                endingRowIndex = min(imageRowIter+segmentationStride+segmentationOverlap,inputImageShapeRow)
                startingColIndex = 0
                endingColIndex = segmentationStride+segmentationOverlap
            else:
                startingRowIndex = imageRowIter-segmentationOverlap
                endingRowIndex = min(imageRowIter+segmentationStride+segmentationOverlap,inputImageShapeRow)
                startingColIndex = imageColIter-segmentationOverlap
                endingColIndex = min(imageColIter+segmentationStride+segmentationOverlap,inputImageShapeCol)

            data1 = inputImage[startingRowIndex:endingRowIndex,startingColIndex:endingColIndex,:]
            #print(data1.shape) 


            patchToProcessData1= np.copy(data1)
            inputToNetData1=torch.from_numpy(patchToProcessData1).type(torch.cuda.FloatTensor)
            inputToNetData1 = inputToNetData1.permute(2,0,1)
            inputToNetData1 = torch.unsqueeze(inputToNetData1,0)


            if useCuda:
                inputToNetData1 = inputToNetData1.cuda()    


            ##Obtaining projection
            model.eval()
            model.requires_grad=False
            with torch.no_grad():
                projection1,_ = model(inputToNetData1,inputToNetData1) 



            _, prediction1 = torch.max(projection1,1)  

            ##Obtaining segmentation maps
            prediction1Squeezed = torch.squeeze(prediction1).cpu().numpy().astype(int)

            prediction1Squeezed = filters.rank.modal(prediction1Squeezed,disk(3))

            if imageRowIter==0 and imageColIter==0: 
                segmentationMap1[0:segmentationStride,0:segmentationStride] = prediction1Squeezed[0:segmentationStride,0:segmentationStride]
            elif imageRowIter==0:
                segmentationMap1[0:segmentationStride,imageColIter:min(imageColIter+segmentationStride,inputImageShapeCol)] =\
                                                                 prediction1Squeezed[0:segmentationStride,\
                                                                                    segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[1])]   
            elif imageColIter==0:
                segmentationMap1[imageRowIter:min(imageRowIter+segmentationStride,inputImageShapeRow),imageColIter:min(imageColIter+segmentationStride,inputImageShapeCol)] =\
                                                                 prediction1Squeezed[segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[0]),\
                                                                                      0:segmentationStride] 
            else:
                segmentationMap1[imageRowIter:min(imageRowIter+segmentationStride,inputImageShapeRow),imageColIter:min(imageColIter+segmentationStride,inputImageShapeCol)] =\
                                                                  prediction1Squeezed[segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[0]),\
                                                                                       segmentationOverlap:min(segmentationOverlap+segmentationStride,prediction1Squeezed.shape[1])]   


    selectedSegmentationMap1 = segmentationMap1 
    selectedSegmentationMap1Color = convert_to_color(selectedSegmentationMap1,palette)
 

    return selectedSegmentationMap1Color


def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d