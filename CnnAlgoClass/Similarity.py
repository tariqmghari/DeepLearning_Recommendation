import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def extract_imgsFeatures(model, data_loader,GPU=False):
    images_features = []

    for image, label in data_loader:
        if(GPU):
            image=image.to('cuda')
            label = label.to('cuda')
        feature = model.extract_features(image)
        images_features.append(feature[0].tolist())
    
    return np.array(images_features)

def product_similatity(images_features, imgsFile_path, csvFileName=None):

    cosSim = cosine_similarity(images_features)

    imagesNames = imgs_names(imgsFile_path)
    df_cosSim = pd.DataFrame(cosSim, columns=imagesNames, index=imagesNames)
    if csvFileName is not None:
        csvFile = "similarity_csv/"+csvFileName+".csv"
        df_cosSim.to_csv(csvFile)
        print("File saved successfully in "+csvFile)

    return df_cosSim

def imgs_names(file_path):
    return [x for x in os.listdir(file_path) if "png" in x]

def show_image(imagePath):
    img = Image.open(imagePath)
    plt.imshow(img)
    plt.show()

def similar_products(df_cosSim, imageName, n):
    imgsFile_path = "Data/style/"
    plt.title("Original product",color='b')
    show_image(imgsFile_path + imageName)

    # most similar products
    mostPrd_Name = df_cosSim[imageName].sort_values(ascending=False)[1:n+1].index
    mostPrd_scores = df_cosSim[imageName].sort_values(ascending=False)[1:n+1]

    for i in range(n):
        plt.title("similarity score :"+str(mostPrd_scores[i]), color='b')
        show_image(imgsFile_path + mostPrd_Name[i])
