"""
A little script to realize blurring of picture data for experimentation,
which is either started by communication client within the CoNM environment or manually.
It realizes experiments on continual ANN training and testing on switching datasets.
Copyright (c) 2022 Marcus Grum
"""

__author__ = 'Marcus Grum, marcus.grum@uni-potsdam.de'
# SPDX-License-Identifier: AGPL-3.0-or-later or individual license
# SPDX-FileCopyrightText: 2022 Marcus Grum <marcus.grum@uni-potsdam.de>

import os
import skimage.io
import matplotlib.pyplot as plt
import skimage.filters # pip3 install scikit-image

def realize_GaussianBlurring(pathInputPictures, pathOutputPictures, sigma):

    # Check if directory for blurred images exists
    if not (os.path.exists(pathOutputPictures)):
        # Create a new directory because it does not exist 
        os.makedirs(pathOutputPictures)
    
    # Parse all images and carry out blurrying
    print("Blurring images specified in path '" + pathInputPictures + "'...")
    for filename in os.listdir(pathInputPictures):
        f = os.path.join(pathInputPictures,filename)
        if os.path.isfile(f):

            # collect relevant file information
            currentFilePath = f
            currentFileName = os.path.basename(f)

            # filter irrelevant file types
            if(currentFileName != ".DS_Store"):

                # load current image
                currentImage_original = skimage.io.imread(fname=currentFilePath)

                # if required, display the original image
                #fig, ax = plt.subplots()
                #plt.imshow(image)
                #plt.show()

                # apply Gaussian blur for creating a new image
                currentImage_blurred = skimage.filters.gaussian(currentImage_original, sigma=(sigma, sigma), truncate=3.5, channel_axis=2)

                # display blurred image
                fig, ax = plt.subplots()
                plt.imshow(currentImage_blurred)
                plt.axis('off')
                #plt.show()

                # storing of blurred image
                fig.savefig(pathOutputPictures + currentFileName, bbox_inches='tight', pad_inches = 0, transparent=True)
                plt.close()

        else:
            print('  directories:', f)

if __name__ == '__main__':
    
    sigma = 6

    # when script is started manually, initiate blurring agent
    # - for apple data
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/train/apple-ok/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/train/apple-ok/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/train/apple-def/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/train/apple-def/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/validation/apple-ok/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/validation/apple-ok/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/validation/apple-def/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/validation/apple-def/", sigma)

    # - for banana data
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/train/banana-ok/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/train/banana-ok/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/train/banana-def/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/train/banana-def/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/validation/banana-ok/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/validation/banana-ok/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/validation/banana-def/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/validation/banana-def/", sigma)

    # - for orange data
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/train/orange-ok/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/train/orange-ok/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/train/orange-def/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/train/orange-def/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/validation/orange-ok/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/validation/orange-ok/", sigma)
    realize_GaussianBlurring("./../../data/fruits-fresh-and-rotten-fruits-dataset/validation/orange-def/", "./../../data/fruits-fresh-and-rotten-fruits-dataset-blurredWithSigma"+str(sigma)+"/validation/orange-def/", sigma)

    # comment out to keep current results and avoid accidental activation
    #pass