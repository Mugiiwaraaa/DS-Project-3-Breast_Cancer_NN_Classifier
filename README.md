# DS-Project-3-Breast_Cancer_NN_Classifier
Classifying whether the tumor is 'Benign' or 'Malignant' . 

* Designed a model that predicts the chances of a patient having **Breast Cancer**. This is a small step towards how we can detect early symptoms of Breast Cancer which gives us a better chance of prevention.
* Dataset - This particular dataset can be imported from the sklearn.datasets library or you can download it from the kaggle resource given below.
* Model - The major aim in this project is to predict the type of tumor(Malignant or Benign) based on the features using some of the deep learning techniques and algorithms. Designed , Compiled and Trained a Neural-Network to classify the type of tumor.

## Code and Resources Used ##
**Python Version:** 3.10.5 <br />
**Packages:** pandas, pandas-profiling, numpy, sklearn, matplotlib, tensorflow <br />
**For Web Framework Requirements:** _pip install -r requirements.txt_ <br />
**Data Resources:** <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>

## About the Dataset ##
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. <br />
The attributes which we recognize as the features of this model are as follows : <br />
1. **ID number**
The attributes which we recognize as the features of this model are as follows : <br />
2. **Diagnosis** (M = malignant, B = benign) - This is the label of the model. Output data that we use to classify the tumor which gives us indications of Breast Cancer. <br />

**Ten real-valued features are computed for each cell nucleus:**

a) **radius** (mean of distances from center to points on the perimeter) <br />
b) **texture** (standard deviation of gray-scale values) <br />
c) **perimeter** <br />
d) **area** <br />
e) **smoothness** (local variation in radius lengths) <br />
f) **compactness** (perimeter^2 / area - 1.0) <br />
g) **concavity** (severity of concave portions of the contour) <br />
h) **concave points** (number of concave portions of the contour) <br />
i) **symmetry** <br />
j) **fractal dimension** ("coastline approximation" - 1) <br />

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in **30 features**. <br />
For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

## EDA - Exploratory data analysis ## 
You can find all related EDA in my Github repository. <br />
Apart from the basic Data analysis which includes pandas function such as: <br />
1. value_counts for the label <br /> 
2. Statistical insight using the describe functions <br /> 
3. finding strong postive and negative correlations using the corr() function <br />
You can check out the extensive EDA report for this dataset which I have generated using the pandas-profiling library. (This is a ninja technique :P) <br />
[ Note: High chances of lags and hangs for large datasets if you are on a potato just like me :) ]


## Model Building ##
I also split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.

I tried three different models:

Multiple Linear Regression – Baseline for the model
DecisionTree Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like DecisionTree would be effective.
Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.

## Model Performance ##
The Random Forest Regression model outperformed the other approaches on the test and validation sets.I found out the Mean and the Standard Deviation for all the three models. This basically gives us an image by how much percentage can the predictions be off the actual prediction.
* **Multiple Linear Regression:**   Mean- 5.028337074958086 , Standard Deviation- 1.056869119278954
* **Decision Tree Regression:** Mean- 4.256820741921791,
   Standard Deviation- 1.1575140416039331
* **Random Forest Regression:** Mean- 3.304827981052571, 
   Standard Deviation- 0.6490112395533792 <br />
The results were pretty good since in the worst case scenario,our predictions would only be off by a meager 3.65%.
