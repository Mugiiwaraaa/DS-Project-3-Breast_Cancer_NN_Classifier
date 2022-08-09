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
1. value_counts for the label attribute <br /> 
2. Statistical insight using the describe functions <br /> 
3. finding strong postive and negative correlations using the corr() function <br />
You can check out the extensive EDA report for this dataset which I have generated using the pandas-profiling library. (This is a ninja technique :P) <br />
[ Note: High chances of lags and hangs for large datasets if you are on a potato just like me :) ]


## Model Building ##
I split the data into train and tests sets with a test size of 20%.

Then using the **StratifiedShuffleSplit** function from the sklearn.model_selection library, I make sure there are similar ratios of the ***Diagnosis*** column in both the test and train data sets.

After shuffling the data, I seperated the diagnosis column as the target/label obtaining features and labels of the dataset.

Finally,before using the train data to train the and build the neural network, I standardize the features by using the StandardScaler function.

#### Setting up layers for the neural network ####

After Initializing the neural network model,I define 3 layers for my model as follows:
1. 1st layer as the **input** layer - This layer consists of 30 neurons since we have 30 features for our dataset. Important to Flatten it since it requires a single dimension array  
2. the middle **hidden** layer - Can have random number of neurons and use the **'ReLU'** activation function since I dont want to activate all the neurons at the same time. 
3. Finally, the **output** layer - Number of neurons must be equal to the number of output classes. In our case this is **2** since we want to classify the tumor either as 'Benign' or 'Malignant'. For my activation function I use the **Sigmoid function** which is th esame as Logistic Regression.
[Note: You can do more research on different activation functions and also play around with the number of layers. Although this might make your model more susceptible to overfitting.]    
#### Compiling the neural network  ####

After setting up the layers, I compiled my model as follows:
* For the Optimizer I pick the **'adam's optimization algorithm** as it uses the principle of stochastic gradient descent.
* For my loss function I chose **sparse_categorical_crossentropy** because the labels are labeled as integers.
* Finally I choose **accuracy** for metrics.



## Model Performance ##
The Random Forest Regression model outperformed the other approaches on the test and validation sets.I found out the Mean and the Standard Deviation for all the three models. This basically gives us an image by how much percentage can the predictions be off the actual prediction.
* **Multiple Linear Regression:**   Mean- 5.028337074958086 , Standard Deviation- 1.056869119278954
* **Decision Tree Regression:** Mean- 4.256820741921791,
   Standard Deviation- 1.1575140416039331
* **Random Forest Regression:** Mean- 3.304827981052571, 
   Standard Deviation- 0.6490112395533792 <br />
The results were pretty good since in the worst case scenario,our predictions would only be off by a meager 3.65%.
