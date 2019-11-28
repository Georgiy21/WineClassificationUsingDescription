import pandas as pd
import numpy as np

class Wine:
    def __init__(self, country, description, province, variety):
        self.information = str(country) + ', ' + str(description) + ', ' + str(province)
        self.variety = variety

initial_data = pd.read_csv('Wine Reviews Project\\train_data.csv', encoding='utf-8')
# print("Data before duplicates are removed: " , len(initial_data)) #* Data before duplicates are removed:  49975

data = initial_data[initial_data.duplicated('description', keep=False)]
# print("Data after duplicates are removed: " , len(data)) #* Data after duplicates are removed:  10102

# Creating a list of Wine objects where attribute "information" consists of country, description 
# and province, and attribute variety is the variety
classif = []
for index, c in data.iterrows():
    classif.append(Wine(c['country'], c['description'], c['province'], c['variety']))

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
training, test = train_test_split(classif, test_size=0.1, random_state=42)

# Splitting the training set into features and target
train_x = [x.information for x in training]
train_y = [x.variety for x in training]

# Splitting the testing set into features and target
test_x = [x.information for x in test]
test_y = [x.variety for x in test]

""" Count Vectorizer """
from sklearn.feature_extraction.text import CountVectorizer
c_vectorizer = CountVectorizer()
train_x_v = c_vectorizer.fit_transform(train_x)
test_x_v = c_vectorizer.transform(test_x)

""" TF-IDF Vectorizer """
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vectorizer = TfidfVectorizer()
Train_x_v = tf_vectorizer.fit_transform(train_x)
Test_x_v = tf_vectorizer.transform(test_x)

''' CLASSIFICATION '''

""" Linear SVM """ # exited with code=1 in 1213.782 seconds
# !Takes forever to produce an output, so discard it
# from sklearn import svm
# classif_svc = svm.SVC(gamma='auto')
# classif_svc.fit(train_x_v, train_y)
# classif_svc.predict(test_x_v)

# print("\\nMean accuracy of SVC using Count Vectorizer:")
# print(classif_svc.score(test_x_v, test_y))

# classif_svc.fit(Train_x_v, train_y)
# classif_svc.predict(Test_x_v)
# print("\\nMean accuracy of SVC using TF-IDF Vectorizer:")
# print(classif_svc.score(Test_x_v, test_y))

""" Stochastic Gradient Descent """
from sklearn.linear_model import SGDClassifier
# classif_sgd = SGDClassifier(loss="hinge", penalty="l2", shuffle=True) 
# classif_sgd.fit(train_x_v, train_y)
# classif_sgd.predict(test_x_v)
# print("Mean accuracy of SGD using Count Vectorizer:")
# print(classif_sgd.score(test_x_v, test_y))

# classif_sgd.fit(Train_x_v, train_y)
# classif_sgd.predict(Test_x_v)
# print("Mean accuracy of SGD using TF-IDF Vectorizer:")
# print(classif_sgd.score(Test_x_v, test_y))

#? Output before removing duplicates (49975 samples):
# Mean accuracy of SGD using Count Vectorizer:
# 0.7322929171668667
# Mean accuracy of SGD using TF-IDF Vectorizer:
# 0.7030812324929971

#? Output after removing duplicates (10102 samples):
# Mean accuracy of SGD using Count Vectorizer:
# 0.9574678536102869 #? <-- the best (SGD) 
# Mean accuracy of SGD using TF-IDF Vectorizer:
# 0.9188921859545005

""" Radial Basis Function Kernel """ 
#! ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
# from sklearn.kernel_approximation import RBFSampler
# classif_rbf = RBFSampler(gamma=1, n_components=20990, random_state=1)
# features_train_x_v = classif_rbf.fit_transform(train_x_v)
# clf = SGDClassifier()
# clf.fit(features_train_x_v, train_y)
# clf.predict(test_x_v)

# print("Mean accuracy of SGD with RBF kernel using Count Vectorizer:")
# print(clf.score(test_x_v, test_y))

# clf.fit(Train_x_v, train_y)
# clf.predict(Test_x_v)
# print("Mean accuracy of RBF using TF-IDF Vectorizer:")
# print(clf.score(Test_x_v, test_y))

""" Decision Tree """
from sklearn.tree import DecisionTreeClassifier
# classif_dec = DecisionTreeClassifier()
# classif_dec.fit(train_x_v, train_y)
# classif_dec.predict(test_x_v)

# print("Mean accuracy of Decision Tree Clissifier using Count Vectorizer:")
# print(classif_dec.score(test_x_v, test_y))

# classif_dec.fit(Train_x_v, train_y)
# classif_dec.predict(Test_x_v)
# print("Mean accuracy of Decision Tree Clissifier using TF-IDF Vectorizer:")
# print(classif_dec.score(Test_x_v, test_y))

#? Output before removing duplicates (49975 samples):
# Mean accuracy of Decision Tree Clissifier using Count Vectorizer:
# 0.6818727490996399
# Mean accuracy of Decision Tree Clissifier using TF-IDF Vectorizer:
# 0.6634653861544618

#? Output after removing duplicates (10102 samples):
# Mean accuracy of Decision Tree Clissifier using Count Vectorizer:
# 0.9515331355093967
# Mean accuracy of Decision Tree Clissifier using TF-IDF Vectorizer:
# 0.9475766567754699

""" Naive Bayes """
# !This model requires bigger array size for my data, so skip it
# from sklearn.naive_bayes import GaussianNB
# classif_na = GaussianNB()
# classif_na.fit(train_x_v.toarray(), train_y.toarray())
# print(classif_na.predict(test_x_v[0]))

""" Logistic Regression """
from sklearn.linear_model import LogisticRegression
# classif_log = LogisticRegression()
# classif_log.fit(train_x_v, train_y)
# classif_log.predict(test_x_v)

# print("Mean accuracy of Logistic Regression using Count Vectorizer:")
# print(classif_log.score(test_x_v, test_y))

# classif_log.fit(Train_x_v, train_y)
# classif_log.predict(Test_x_v)
# print("Mean accuracy of Logistic Regression using TF-IDF Vectorizer:")
# print(classif_log.score(Test_x_v, test_y))

#? Output before removing duplicates (49975 samples):
# Mean accuracy of Logistic Regression using Count Vectorizer:
# 0.7486994797919168 #? <-- the best (Logistic Regression)
# Mean accuracy of Logistic Regression using TF-IDF Vectorizer:
# 0.6860744297719088

#? Output after removing duplicates (10102 samples):
# Mean accuracy of Logistic Regression using Count Vectorizer:
# 0.9525222551928784
# Mean accuracy of Logistic Regression using TF-IDF Vectorizer:
# 0.6983184965380811

''' Saving Model '''
import pickle

with open('./Wine Reviews Project/wine_classifier_sgd.pkl', 'wb') as f:
    pickle.dump(classif_sgd, f)

#TODO:      NOW LET'S TAKE OUR BEST MODELS 
#TODO:      AND TEST THEM ON A BIGGER DATASET.
#TODO:      DISPLAY RESULT AND ACTUAL DATA FOR VISUAL COMPARISON     

initial_test_data = pd.read_csv('Wine Reviews Project\\train_data - Copy.csv', encoding='utf-8')
test_data = initial_test_data.dropna(subset=['description', 'country', 'province'])

# print("initial_test_data: " , len(initial_test_data)) #* 150930
# print("test_data: " , len(test_data)) #* 150925

test_set = []
for index, c in test_data.iterrows():
    test_set.append(Wine(c['country'], c['description'], c['province'], c['variety']))

new_training, new_test = train_test_split(test_set, test_size=.8, random_state=42, shuffle=False)

new_test_x = [x.information for x in new_test]
new_test_y = [x.variety for x in new_test]

c_testing = CountVectorizer()
train_x_v = c_testing.fit_transform(train_x)
new_test_x_v = c_testing.transform(new_test_x)

tf_testing= TfidfVectorizer()
Train_x_v = tf_testing.fit_transform(train_x)
new_Test_x_v = tf_testing.transform(new_test_x)

new_classif_sgd = SGDClassifier(loss="hinge", penalty="l2", shuffle=True) 
# new_classif_sgd.fit(train_x_v, train_y)
# new_classif_sgd.predict(new_test_x_v)
# # print("Mean accuracy of SGD using Count Vectorizer:") #* 0.5213019711777372
# # print(new_classif_sgd.score(new_test_x_v, new_test_y)) 

new_classif_sgd.fit(Train_x_v, train_y)
new_classif_sgd.predict(new_Test_x_v)
# # print("Mean accuracy of SGD using TF-IDF Vectorizer:") #* 0.5800066258075203 <-- the best
# # print(new_classif_sgd.score(new_Test_x_v, new_test_y))

new_classif_log = LogisticRegression()
new_classif_log.fit(train_x_v, train_y)
new_classif_log.predict(new_test_x_v)
# # print("Mean accuracy of Logistic Regression using Count Vectorizer:") #* 0.5602865661752526 <-- the best
# # print(new_classif_log.score(new_test_x_v, new_test_y))

# new_classif_log.fit(Train_x_v, train_y)
# new_classif_log.predict(new_Test_x_v)
# # print("Mean accuracy of Logistic Regression using TF-IDF Vectorizer:") #* 0.5000745403346033
# # print(new_classif_log.score(new_Test_x_v, new_test_y))

''' Display for compariosn '''

print("PRINTING LAST 10 ROWS OF TESTING SET:")
print(test_data.tail(n=10))
print("PREDICTING VARIETY WITH SGDClassifier (TF-IDF VECTORIZER): ")
print(new_classif_sgd.predict(new_Test_x_v[10:]))
print(" ")
print("PREDICTING VARIETY WITH LogisticRegression (Count Vectorizer): ")
print(new_classif_log.predict(new_test_x_v[10:]))