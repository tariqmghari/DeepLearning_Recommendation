# Recommendation System

<span style='color:red'>NOTE: This project does not contain all files, which means non functional. (Company Copyright)</span>

### Introduction: 

The goal of this project is to first train a model that will be able to detect similarities between products using image recognition techniques (Deep learning) in order to recommend articles based on the physical similarity.

Dataset used : https://www.kaggle.com/olgabelitskaya/style-color-images 

Different neural network models will be used to find the right model for our project.

**Tools used: ** Pytorch for Neural network, scikit-learn for clustering. 

### PLAN:

```
1. Data collection
2. Data preparation
3. Model building (CNN)
4. Model training
5. Data augmentation
6. Model training with augmented data
```

### Data Collection:

The data I have used for this project, is a public dataset that contains products and some information.

> The main dataset (style.zip) is 2184 color images (150x150x3) with 7 brands and 10 products, and the file with labels `style.csv`.
> Photo files are in the `.png` format and the labels are integers and values.
>
> The file `StyleColorImages.h5` consists of preprocessing images of this set: image tensors and targets (labels).  
>
> ~~Kaggle dataset description

### Data Preparation:

Preparing a class in order to import data and transform it for the model used. [LoadData()][Ref] 

[Ref]: https://github.com/tariqmghari/DeepLearning_Recommendation/blob/master/CnnAlgoClass/LoadData.py	"LoadData()"

### Model Building:

Our goal is not image classification, but its image similarities, and to do so we do not require the classification layers in the end of the model, so instead of taking the whole model we substract the last classification layer, this will leave us with the output of the layer before the classification which is a vector of values (This process is called feature extraction)

> Feature extraction is **a type of dimensionality reduction where a large number of pixels of the image are efficiently represented in such a way that interesting parts of the image are captured effectively**.
>
> ~~ScienceDirect.com

The vector we get after passing the images through the model is called image features, and we can use these vectors to calculate the distance between each vector using, **Euclidean Distance, Cosine ...**, in my case I have used cosine.

##### <u>VGG19:</u>

![](https://www.researchgate.net/profile/Clifford-Yang/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg)

**<u>Alexnet:</u>**

![](https://neurohive.io/wp-content/uploads/2018/10/AlexNet-1.png)

**<u>Resnet:</u>**

![](https://miro.medium.com/max/1400/0*9LqUp7XyEx1QNc6A.png)



I have worked on different architectures in order to find the right model for my case.

**<span style='color:red'>PS: All the models I have used are all pretrained on [ImageNet](www.image-net.org) dataset, with 14m images that were tagged into ~20 000 categories</span>**

### Data augmentation:

With only 2000 images, the model performance was a bit low, especially when we use new pictures with different angles.

> Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.
>
> ~Wikipedia

To augment the data we apply a transform on each picture and save it along with our initial dataset.

```python
tf = transforms.Compose([
    transforms.ToTensor(),# Transforming picture into a tensor
    transforms.ToPILImage(),
    transforms.Resize((170, 170)),#resizing picture for model input
    transforms.RandomCrop((150, 140)),# we crop teh picture randomly to make some noise on the picture
    transforms.ColorJitter(brightness=0.5),#change brightness
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])
```

### Conclusion:

To conclude, to make a recommendation system based on image content, the steps are:

- Gather data
- Create CNN models and remove the last classification layer
- Take the output for each picture and calculate the cosine distance of the vector to all other vectors
- The cosine value is 0 to 1, 0 means dissimilar, 1 similar