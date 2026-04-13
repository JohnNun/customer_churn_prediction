# Customer Churn Prediction
[](images/1981-digital-yqaskj8lQBE-unsplash-copy.jpg)
###### photo by [1981 Digital](https://unsplash.com/@1981digital) on [Unsplash](https://unsplash.com)
Customer retention is essential for any live-service and subscription-based business models as this greatly impacts company revenue and growth. When a product or service fails to meet customer expectations or an alternative becomes more compelling, customers may discontinue their subscription service with the company, creating customer churn. The focus of this repository is the development and training of a machine learning algorithm that can predict if a customer will churn based off user usage data. The data used for model training is the Customer Churn Dataset found on Kaggle.com, uploaded by Kaggle user Muhammad Shahid Azeem. In this dataset there are over 500k rows of customer data usage with each record including features such as customer age, gender, tenure, usage frequency, support calls, payment delay, subscription type, contract length, total spend, and last interaction. The data also contains a churn label feature that indicates whether the customer has churned (1) or not (0). In this repository, the final model used is a Random Forest Classifier as this model outperformed all other models at the base level, with a CNN model being a close second. Various evaluation metrics highlights the model’s performance: for “not churn,” the model achieved high precision (0.99) but lower recall (0.86) and an F-score of 0.92; for “churn,” the model scored a precision of 0.90, a perfect recall (1.00), and an F-score of 0.94. Despite heavy data imbalance, the model is able to reliably predict customer churn and retention, that said, further fine-tuning and data balancing could possibly further enhance overall model performance and predictive capabilities.

# Business Understanding
For a business with a live service product or a subscription-based product, it often requires active consumer retention to maintain the product or service running. In order for a business to maintain consumer retention they often have to produce a product that will set them apart from any others that offer the same kind of service or product. In some cases, for whatever reason the offered product can fail to retain consumers leading to customer churn.

Customer churn refers to when customers discontinue their relationship or subscription with a company or service. It also represents the rate at which customers stop using a company's products or services within a specific period.
Understanding customer churn is crucial for businesses as this lets them identify patterns, factors, and indicators that contribute to customer attrition. Churn is an important business metric as it impacts revenue, and growth. By analyzing churn behavior and its associated features, companies can develop strategies to retain existing customers, improve customer satisfaction, and reduce customer turnover. Through the use of predictive machine learning models, it is possible to forecast and proactively address potential churn, allowing companies to take the necessary measures to minimize or even retain at risk consumers.

# Data Understanding
The data used for model training is the [Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset) found on [Kaggle.com](https://www.kaggle.com/), uploaded by Kaggle user [Muhammad Shahid Azeem](https://www.kaggle.com/muhammadshahidazeem). In this dataset we can find over 500k rows of customer data usage with each record including features such as age, gender, tenure, usage frequency, support calls, payment delay, subscription type, contract length, total spend, and last interaction. The data also contains a churn label that indicates whether the customer has churned (1) or not (0).

Upon first looks of the data we can tell the data is heavily imbalanced with "churn" having 55k+ more rows of data compared to it "not churn". The data also appears to be very clean with only 1 NaN row which seems to just be a filler row of some kind. As for the data types, the categorical features are the proper dtype, but the numerical features are a mixture of integer and float dtypes.

## Data Preprocessing
Prior to preparing the data for model training I did some minor data cleaning as luckily this data is very clean and organized. First step I took for data preprocessing was to concat both the train data and test data into a single dataset as both these data sets are imbalanced and by combining both these sets it will help get a better data split when it comes to splitting between train and test data. After combining the datasets, I make a copy of the data to avoid any potential row change warnings and drop the single NaN row as well as reset the data’s index since I combined two datasets together. From there I lowercase the feature titles and replace any whitespaces with underscores, I lastly convert most of the numerical values to integer data type as all except the 'total spend' feature values are whole numbers, and by doing so I can keep the numerical data types uniform.

## Exploratory data Analysis

To understand the data a bit better and the way it is distributed, I did a little bit of Exploratory Data Analysis (EDA) with some of the features in the data.

The box plot that shows how spread out the data is with payment delay and customer churn. From this plot we can see that customers with shorter delays on payment were less likely to churn, while those with a delay of 10+ days had a much larger chance of ending their service.

The second plot is a simple bar plot that shows us the age distribution in the data. From this plot we can see that the majority of customers are between the ages of 18 - 50 with the greater majority being between 40 - 50 years of age, and ages 51 - 65 being the minority age group. The third plot is a continuation of the second plot as this plot shows us age to churn ratio. From this plot we can see customers between the ages of 30 - 50 are less likely to end service, while those between the ages of 18 - 29 and 51 - 65 are more likely to churn.

The final plot shows the "churn" to "not churn" frequency amongst genders found in the dataset. This plot shows us that between male and female customers, female customers are much more likely to churn while male customers are less likely to churn.
[](images/boxplt_payment_churn_dist.jpeg)
[](images/barplt_age_distribution.jpeg)
[](images/countplt_age_churn_dist.jpeg)
[](images/countplt_gender_churn_dist.jpeg)

## Data Preperation
To prepare the data I first set my X and y variables, for the X variable I kept all features except for the “churn” feature as this is our y variable, and “customerid” as this feature is a unique identifier and essentially holds no value. For the train and test split I split the data to be 80% as training data and 20% as test data, I also set the random state seed to 42 for consistent results for each run of the notebook. After the split I create copies of the training and testing sets of data to use for pipeline development. To finalize the data preparation, I use two functions from the Flatiron school to encode the categorical features found in the data and transforms a feature into multiple columns of 1s and 0s, I then create a function that concats these new features into our dataframe while dropping the old categorical feature columns.

# Modeling
For the modeling process I created a total of 4 models 2 supervised, 2 deep learning, to gauge which model would work best with the data at a base level and later fine tune. The two supervised learning models I used were Logistic Regression and Random Forest Classifier, I went with these two supervised learning models as logistic regression models work well in predicting binary outcomes, in this case "churn" or "no churn". I also chose a Random Forest Classifier as a test model as this machine learning model is fairly accurate with their prediction, and so are often used when it comes to churn predictions.

For the deep learning base models, I went with a Sequential model as well as a Convolutional Neural Network (CNN) model. I chose a Sequential model as same as the random forest; these kinds of models do well with data that can be used for forecasting customer retention. A Convolutional Neural Network can be seen as a weird choice for a test model as these kinds of models are primarily meant for image recognition. That being said CNN models are also good at recognizing patterns in data early on as well as complex patterns in the deeper layers.

Ultimately, the final model used was Random Forest Classifier as this model outperformed all other models at the base level, with the CNN model being a close second. To fine tune the random forest model I used GridSearchCV to find the best parameters from a range of possible settings, and the best parameters for the rf classifier were a gini criterion with no max depth, balanced subsampling, 1 minimum samples per leaf, and 5 for minimum samples before a split.

Please note the Gridsearch process has been commented out as it takes more than 2 hours to run even with Google Colabs Pro+ resource privileges.

# Evaluation
The final model performed fairly well overall, but it does appear to struggle a bit when it comes to predicting if a customer is "not churn". This is **most likely due to the heavy imbalance** in the data creating some **model bias**, or it is possible that some of the "not churn" rows of data have very similar pattern to the "churn" classification, but this seems unlikely.

For the model’s classification metrics, it scored a **precision of 0.99, recall of 0.86, and an f-score of 0.92** when it came to predicting "not churn". For predicting customer "churn" the model scored a **precision of 0.90, a recall of 1.00, and an f-score 0.94**. From this report we can see that while the model may be precise with its "not churn" prediction, it does **struggle to recall** what it learned from this kind of data. For its customer "churn" predictions on the other hand, the model has a **lower precision rate** but it is able to recall all that it has learned from this kind of data.

By using a confusion matrix, we are able to get a visual representation of the classification scores as well as how the model did with its predictions. For "not churn" the model was able to **correctly predict over 38k customer** status with 6507 the model believing were "churn" when in fact were not. For "churn" the model did much better with **over 55k correct predictions** and only 205 misclassified predictions.

## Limitations
The biggest limitation for this model is the heavy imbalance in the data with **over 55k** more rows of data in favor of the "churn" classification as compared to "not churn".

A second limitation I encountered was with the Random Forest Classifier and it's **limited parameter settings** when it comes to dealing with the imbalanced data set and the more balance I tried to do, the model would get lower scores overall.

## Next Steps
A potential next step would be to **incorporate SMOTE** into the data to create enough fake data to balance out the data set, but the issue with using SMOTE is that it would be **making up the data** and in a real world situation using **real customer data would be much more beneficial**.

A second next step would be to **fine-tune the second best model** in this notebook which is the CNN model, compared to the random forest model, the **CNN model underperformed by about 2 points** in the validation scores, so with a difference that low it would be something **worth looking into for better results**.

# Conclusion
Overall, the model performed very well despite the heavy class imbalance found in the data. The model is able to correctly predict the majority of the data used during testing and with deeper fine-tuning and data balancing, it may be possible to yield better overall results lowering the model’s false negative results.

## Repository Structure
* images
* README.md
* [Notebook](customer_churn_prediction.ipynb)
