
# Introduction
## 1. Classification (using Logistic classfication)
In this small project, **classification** using **big data** tool (e. g., **pyspark**) is deployed. The data for exploration and classification is derived from the very popular source. [cencus income dataset from UCI machine learning](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/)
- The google colab is used to implement to classification. Then, in the first part, the pyspark with **hadoop** is setup on the google cloud.
- In the second part, data is reading using pyspark as a dataframe that is then used to explore features and analyse by using **logistic classification**. There are two parts of data, the train data and test data that are consistent and representative. 
- Data obtained in the previous step will explore to see how relevant between features. 
- Data indeed are either continuous or categorical. Then, in order to be used as input of classifier, those data need to be converted into numerical so that the classifier can understand. **StringIndexer** and **OneHotEncoderEstimator** are used to convert **nominal categorical features** into the understandable format.
- The **Pipeline process** has been deploy to reduce the manually repeated the same process over time. Indeed, there are many categorical features need to be indexed and converted to numerical format, and **VectorAssembler** is used to combine all necessary features as a single feature to be inputted into classifier.
- Next step, The **logistic classification** is utilized to sucessfully classify observations from dataset into correct class. In this small project, there are only two classes, income higher than 50K and less than 50K. The classifier is set so that its hyperparameter **regularization parameter** is equal to 0.2, this a a random value to set and is tuned later to get the optimal value for this hyper parameter. 
- In addition, evaluation step is used to confirm the effectiveness of our classification about the data. Because two classes from the train data is **balance class** with the ratio (40% and 60% for each class), then **accuracy** metric is chosen to evaluate the classification performance. And the logistic classification gave **accuracy = 81%** as compared to the optimal **accucary = 85%** from the author of this dataset. see in the file 'data/adult.names'. Furthermore, the ROC is used to compute the are under the curve, with area 89% very closed to 100%.
- Last step, in order to tune the hyperparameter, the **GridSeaerch** combined with **crossvalidator** are used to find the best hyperparameter for regularization. And the best value is obtained with regularization hyperparameter is 0.01 with accuracy is closed to 84%.        
 
## 2. Anamoly detection (using Kmeans clusters) 
In this small task, **clustering KMeans** using **big data** tool (e. g., **pyspark**) is deployed. The data for exploration and classification is derived from the very popular source. [data from kddcup99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).

- In the first step, the data with 41 attributes are read. Most of the data are double type except 3 attributes are String type. **SparkContext** method are used to read and parse each line into 41 attributes under **Resilient Distributed Dataset**. Then data is converted to again dataframe due to the nominal categorical features that the algorithms can not understand. 
- The **Pipeline process** has been deploy to reduce the manually repeated the same process over time. Indeed, there are many categorical features need to be indexed and converted to numerical format, and **VectorAssembler** is used to combine all necessary features as a single feature to be inputted into classifier.
- The **anomaly dection** means that the data or observations that are different from normal or what we expected. By doing that, statistic method flags the data as 'bad' or 'malicious' when the data behave five times different from the mean of all training data. This method acts on univariate outliers, it is easy to do but it is not really robust because it required the parameters about the distribution, like mean and standard deviation, and it also assumes that the model is normal. In addition, To obtain the **mean** and the **standard deviation**, the **Stats()** funtion is acted on the RDD. However, univariate outliers are not efficient in reality because, multiple actions are often considered to access normal or obnormal. Then, the multivariate should be used, for the sake of the simplicity we do not mention it in this small task.
- Instead, the **Kmeans** is used to visualize the outliers. To verify the dependence of the Kmeans on the data. There are two steps have been done to compare how sensitive KMeans to data. In the first case, the data is not standardized, mean that just use the data with different distribution for all features and let KMeans find the outliers. In the other case, all the features are standardized to ensure that they have the same distribution, indded all the features data are subtracted for their mean and divided by their standard deviation.
- The results show that, for the data without standardize, the KMeans recognizes so many outliers, more than 1000 in cluser 0 and cluster 2. However, when standardizing the data according to all input features, the KMeans gave us only one outliers in cluster 2.  

## 3. Recommendations using ALS

In this small task of recommendation, the dataset is obtained from [dataset for recommendation](https://grouplens.org/datasets/movielens/).

Recommendation is useful to help both provider and consumer to maximize provider's profits or user's knowledge/exploration, and this technique is very popular in big data processing with huge dataset.

In this small task, we will go through following steps based on the idea of matrix factorization [here](https://dl.acm.org/doi/10.1109/MC.2009.263) and collaborative filtering [here](https://ieeexplore.ieee.org/document/4781121).
- load the dataset and processing 
- exploration dataset to find some intuitations about the dataset
- train Alternative Least Square model for recommendation
- make recommendation for the most active user
- evaluation the trained model on test dataset based on metrics rmse, mae. (root mean square error and mean absolute error).
- some conclusions.  
