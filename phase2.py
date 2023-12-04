# -*- coding: utf-8 -*-
"""24100314_Phase2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LKZ75AuCGdrDqzh0wBUqkJeH-ZbSru4y

# CS 535 Machine Learning
## Project Phase 02
\\

- Alina Faisal 24100314

\\

## 0.0 Preprocessing all Datasets

#### 0.1 Extracting the datasets
"""

!pip install unidecode

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from unidecode import unidecode
import re

from google.colab import drive
drive.mount('/content/drive/')

curr_path="/content/drive/MyDrive/ML_ProjP2/"
espn_path = curr_path+"espn_task1.csv"
google_path = curr_path+"Google_task1.csv"
starbucks_path = curr_path+"Starbucks_task1.csv"
ndtv_path = curr_path+"ndtv_task1.csv"
urstrulymahesh_path = curr_path+"urstrulyMahesh_data.csv"
vp_path = curr_path+"VP_task1.csv"

# creating dataframes for each dataset
pd_espn = pd.read_csv(espn_path)
pd_google = pd.read_csv(google_path)
pd_starbucks = pd.read_csv(starbucks_path)
pd_ndtv = pd.read_csv(ndtv_path)
pd_urstrulymahesh = pd.read_csv(urstrulymahesh_path)
pd_vp =  pd.read_csv(vp_path)

pd_espn.rename(columns={'Tweet':'tweet_content'}, inplace=True)
pd_google.rename(columns={'Tweets':'tweet_content'}, inplace=True)
pd_starbucks.rename(columns={'Content':'tweet_content'}, inplace=True)
pd_urstrulymahesh.set_axis(['tweet_content'], axis=1, inplace=True)
pd_vp.set_axis(['tweet_content'], axis=1, inplace=True)

"""#### 0.2 Data Cleaning and Preparation"""

# loading stop words
stop_words_path = curr_path+"stop_words.txt"
stop_words=[]
with open(stop_words_path) as file:
    stop_words.extend(file.read().split('\n'))
stop_words_dict={word:word for word in stop_words}

emoji_pattern = re.compile("["u"\U00002500-\U00002BEF"u"\u231a"u"\u23e9"u"\U0001F600-\U0001F64F"u"\u23cf"u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"u"\U00002702-\U000027B0"u"\u3030"u"\ufe0f"u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251"u"\U00010000-\U0010ffff"u"\u200d"u"\U0001f926-\U0001f937"u"\u2640-\u2642"u"\u2600-\u2B55""]+", flags=re.UNICODE) 

def remove_emojis(string):
  """ Remove emojis from a string

  :param string: string to remove the emojis from
  :type string: string
  :returns: string without the emojis
  :rtype: string """
  
  return emoji_pattern.sub(r'', string)

def clean_text(string):
  """ Clean a given text by removing URLs, stop words, tags, punctuation, accents, and digits
  
  :param string: the text to be cleaned
  :type string: string
  :returns: the cleaned text
  :rtype: string """

  ret_s = []
  splitted = string.split()
  for word in splitted:

    # removing tags
    if word[0]=='@' or word[0]=='_':
      continue

    # removing stop words
    if stop_words_dict.get(word):
      continue
    
    # removing URLs
    if word.startswith("http"):
      continue
    
    # removing single letter words
    if len(word)==1:
      continue
    
    # removing punctuation and special characters
    word = re.sub(r'[!@#$]', '', word)
    temp_word = re.sub('[^\w\s]+',' ', word)

    # removing accents
    temp_word = unidecode(temp_word)

    # removing digits
    temp_word = re.sub(r'[0-9]', '', temp_word)

    ret_s.append(temp_word)

  return " ".join(ret_s)

def data_cleaning(df):
  """ Clean the tweets
  
  :param df: the dataframe containing the tweets
  :type df: pandas DataFrame
  :returns: the dataframe with tweets lowercase, emojis removed, and text cleaned
  :rtype: pandas DataFrame """

  for i,tweet in enumerate(df['tweet_content']):
    df['tweet_content'][i] = df['tweet_content'][i].lower() 
    df['tweet_content'][i] = remove_emojis(df['tweet_content'][i])
    df['tweet_content'][i] = clean_text(df['tweet_content'][i])
  return df

pd_espn['unclean_tweet_content'] = pd_espn['tweet_content']
pd_google['unclean_tweet_content'] = pd_google['tweet_content']
pd_starbucks['unclean_tweet_content'] = pd_starbucks['tweet_content']
pd_ndtv['unclean_tweet_content'] = pd_ndtv['tweet_content']
pd_urstrulymahesh['unclean_tweet_content'] = pd_urstrulymahesh['tweet_content']
pd_vp['unclean_tweet_content'] = pd_vp['tweet_content']

# cleaning all the tweets
pd_espn = data_cleaning(pd_espn)
pd_google = data_cleaning(pd_google)
pd_starbucks = data_cleaning(pd_starbucks)
pd_ndtv = data_cleaning(pd_ndtv)
pd_urstrulymahesh = data_cleaning(pd_urstrulymahesh)
pd_vp = data_cleaning(pd_vp)

"""#### 0.3 Adding labels"""

def give_label_for_handle(df,name):
  """ Add label (author) for each tweet
  
  :param df: the dataframe containing the tweets
  :type df: pandas DataFrame
  :param name: the label for the dataframe
  :type name: string/int/float
  :returns: the dataframe with labels added
  :rtype: pandas DataFrame """
  
  df['twitter_handle'] = df['tweet_content']
  for i,tweet in enumerate(df['tweet_content']):
    df['twitter_handle'][i]=name
  return df

# adding labels
pd_espn = give_label_for_handle(pd_espn,"espn")
pd_google = give_label_for_handle(pd_google,"google")
pd_starbucks = give_label_for_handle(pd_starbucks,"starbucks")
pd_ndtv = give_label_for_handle(pd_ndtv,"ndtv")
pd_urstrulymahesh = give_label_for_handle(pd_urstrulymahesh,"yourstrulymahesh")
pd_vp = give_label_for_handle(pd_vp,"vp")

# merging all datasets after removing extra columns

final_df = pd_ndtv.drop(pd_ndtv.columns[[0,2]], axis=1)
temp_df_google = pd_google.drop(pd_google.columns[[1]], axis=1)
temp_df_starbucks = pd_starbucks.drop(pd_starbucks.columns[[0,1,3]], axis=1)
temp_df_espn = pd_espn.drop(pd_espn.columns[[0,1,3]], axis=1)
temp_df_urstrulymahesh = pd_urstrulymahesh.drop(pd_urstrulymahesh.columns[[1]], axis=1)
temp_df_vp = pd_vp.drop(pd_vp.columns[[1]], axis=1)
final_df = pd.concat([final_df, temp_df_google,temp_df_starbucks,temp_df_espn,temp_df_urstrulymahesh,temp_df_vp], axis=0)

final_df.head()

"""## 1.0 Feature Preparation

#### 1.1 Creating Bag of Words Features

Splitiing the dataset
"""

training_data,test_data = train_test_split(np.array(final_df),test_size=0.2,train_size=0.8,shuffle=True)

training_data = pd.DataFrame(training_data)
test_data = pd.DataFrame(test_data)
training_data.set_axis(['tweet_content', 'tweet_handle'], axis=1, inplace=True)
test_data.set_axis(['tweet_content', 'tweet_handle'], axis=1, inplace=True)

print("Training data shape:",training_data.shape)
print("Test data shape:",test_data.shape)

training_data.head()

# writing out the train and test sets to CSVs
training_data.to_csv('/content/drive/MyDrive/ML_ProjP2/training_data.csv', index=False)
test_data.to_csv('/content/drive/MyDrive/ML_ProjP2/test_data.csv', index=False)

# training_data = pd.read_csv('/content/drive/MyDrive/ML_ProjP2/training_data.csv')
# test_data = pd.read_csv('/content/drive/MyDrive/ML_ProjP2/test_data.csv')

"""Creating Vocabulary"""

def get_counts(tweets):
  """ Get count of each unique token in the dataset of tweets
  
  :param tweets: the tweets from which to obtain counts
  :type tweets: pandas Series / numpy array / python list
  :returns: each unique word in the given tweets against its count
  :rtype: python dictionary """

  v_dict = {}
  for tweet in tweets:
    for word in tweet.split():
      if word not in v_dict.keys():
        v_dict[word] = 1
      else:
          v_dict[word] += 1
  return v_dict

def construct_vocabulary(v_dict):
  """ Create a list of unique words for a given tweets dataset
  
  :param v_dict: all unique words against their counts in the dataset
  :type v_dict: python dictionary
  :returns: a list of unique vocabulary for the tweets
  :rtype: python list """

  v_list=[]
  for key,val in v_dict.items():
      if val>7:
          v_list.append(key)
  return list(set(v_list))

# dropping NULL tweets from the data if any
training_data = training_data.dropna()
test_data = test_data.dropna()

# printing any NULL tweets in the train dataset
for i,row in enumerate(training_data['tweet_content']):
  if type(row)==float:
    print(i,row,training_data['tweet_content'][i],training_data['tweet_handle'][i])

# getting counts of unique words in training data and creating a vocabulary
v_dict = get_counts(training_data['tweet_content'])
v_list = construct_vocabulary(v_dict)

print("Length of vocabulary:",len(v_list))

print("Number of training data tweets:",len(training_data['tweet_content']))
print("Number of test data tweets:",len(test_data['tweet_content']))

"""Creating Bag of Words Representation"""

def bow_representation(tweets,v_list):
  """ Create a bag of words representation for a given dataset of tweets and vocabulary
  
  :param tweets: the tweets to create bag of words for
  :type tweets: pandas Series / numpy array / python list
  :param v_list: the vocabulary for the tweets
  :type v_list: numpy array / python list
  :returns: a dataframe with bag of words for all the tweets
  :rtype: pandas DataFrame
  :returns: a dictionary with counts for all the unique words with add-one smoothing
  :rtype: python distionary """

  len_vocab = len(v_list)
  feature_df = []
  bow_dict={w:1 for w in v_list}
  for tweet in tweets:
    feature_vector=[1 for i in range(len_vocab)]
    words_in_tweet = tweet.split()
    for i,v_word in enumerate(v_list):
      for word in words_in_tweet:
        if word==v_word:
          feature_vector[i]+=1
          bow_dict[v_word]+=1
    feature_df.append(feature_vector)
  return pd.DataFrame(feature_df, columns=v_list), bow_dict

# creating bag of words representation for training and test tweets
bow_df_training, bow_dict_training = bow_representation(training_data['tweet_content'],v_list)
bow_df_test, bow_dict_test = bow_representation(test_data['tweet_content'],v_list)

bow_df_training.head(10)

print("Shape of bag of words for training tweets:",bow_df_training.shape)

print("Counts of some unique tokens in the training tweets with add-one smoothing")
dict((list(bow_dict_training.items()))[:15])

"""#### 1.2 Embeddings"""

!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

training_data['unembedded_tweet_content']=training_data['tweet_content']
test_data['unembedded_tweet_content']=test_data['tweet_content']

# resetting indices
training_data.reset_index(inplace = True, drop = True)
test_data.reset_index(inplace = True, drop = True)

# encoding the tweets in the training and test dataframes
training_data['tweet_content'] = model.encode(training_data['tweet_content'])
test_data['tweet_content'] = model.encode(test_data['tweet_content'])

training_data.head()

training_data['tweet_content'][6]

# creating an array of encodings from the train and test tweets
training_sentences = list(training_data['unembedded_tweet_content'])
training_embeddings = model.encode(training_sentences)

test_sentences = list(test_data['unembedded_tweet_content'])
test_embeddings = model.encode(test_sentences)

print("Shape of training encodings:",(training_embeddings).shape)
print("Shape of test encodings:",(test_embeddings).shape)

"""## 2.0 kNNs"""

!pip install sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict,cross_val_score,cross_val_predict
import seaborn as sns
import matplotlib.pyplot as plt

"""#### 2.1 Preparing Dataset"""

# creating gold labels for the training and test datasets
gold_labels_training = training_data['tweet_handle']
gold_labels_test = test_data['tweet_handle']

# coverting bag of words training and test dataframes to numpy arrays
bow_np = np.array(bow_df_training)
bow_np_test = np.array(bow_df_test)

"""#### 2.1 Helper Functions"""

def skl_find_best_vals(f1_score_e, f1_score_b, accuracy_e, accuracy_b):
  """ Report best k values for kNN classification on embedded and bag of words datasets
  
  :param f1_score_e: all f1 scores for embedded tweets
  :type f1_score_e: numpy array / python list
  :param f1_score_b: all f1 scores for bag of words tweets
  :type f1_score_b: numpy array / python list
  :param accuracy_e: all accuracy values for embedded tweets
  :type accuracy_e: numpy array / python list
  :param accuracy_b: all accuracy values for bag of words tweets
  :type accuracy_b: numpy array / python list
  
  :returns: k values with best accuracy and f1 score on embedded tweets
  :rtype: int """
  
  best_k_f1_e = np.argmax(f1_score_e) +1
  best_k_f1_b = np.argmax(f1_score_b) +1
  best_k_acc_e = np.argmax(accuracy_e) +1
  best_k_acc_b = np.argmax(accuracy_b) +1

  print("Best value for k using f1 score on Embedded features:",best_k_f1_e)
  print("Best value for k using f1 score on Bag of Words features:",best_k_f1_b)
  print("Best value for k using accuracy on Embedded features:",best_k_acc_e)
  print("Best value for k using accuracy on Bag of Words features:",best_k_acc_b)

  return best_k_f1_e, best_k_acc_e

def skl_plotting_function_e_vs_m(vals_k,f1_score_e, f1_score_b, accuracy_e, accuracy_b):
  """ Plot accuracy values and f1 scores for different k values on embedded and bag of words tweets
  
  :param vals_k: values of k in kNN classification
  :type vals_k: numpy array / python list
  :param f1_score_e: all f1 scores for embedded tweets
  :type f1_score_e: numpy array / python list
  :param f1_score_b: all f1 scores for bag of words tweets
  :type f1_score_b: numpy array / python list
  :param accuracy_e: all accuracy values for embedded tweets
  :type accuracy_e: numpy array / python list
  :param accuracy_b: all accuracy values for bag of words tweets
  :type accuracy_b: numpy array / python list

  :returns: k values with best accuracy and f1 score on embedded tweets
  :rtype: int """

  plt.figure(figsize=(30, 20))

  # plotting accuracy values
  plt.subplot(441)
  plt.plot(vals_k, accuracy_e)
  plt.plot(vals_k, accuracy_b)
  plt.xlabel('Values of k')
  plt.ylabel('Classification Accuracy')
  plt.title('Classification Accuracy vs Values of k')
  plt.legend(['Embedding', 'Bag of Words'])
  plt.xlim([1,10])

  # plotting f1 scores
  plt.subplot(442)
  plt.plot(vals_k, f1_score_e)
  plt.plot(vals_k, f1_score_b)
  plt.xlabel('Values of k')
  plt.ylabel('F1 Score')
  plt.title('F1 Score vs Values of k')
  plt.legend(['Embedding', 'Bag of Words'])
  plt.xlim([1,10])

  plt.show()

  print()

  # reporting best k values on both embedded and bag of words tweets with respect to accuracy and f1 score
  best_k_f1_e, best_k_acc_e = skl_find_best_vals(f1_score_e,f1_score_b,accuracy_e,accuracy_b)

  return best_k_f1_e, best_k_acc_e

def plot_confusion_matrix(confmat, title, labels):
  """ Plot confusion matrix
  
  :param confmat: confusion matrix to plot
  :type confmat: numpy array / python list
  :param title: title for the matrix plot
  :type title: string
  :param labels: labels for the plot
  :type labels: numpy array / python list
  :returns: nothing """

  plt.figure(figsize=(10,8))
  ax = sns.heatmap(confmat, linewidth=0.5,xticklabels=labels,yticklabels=labels, annot=True, fmt='g')
  ax.set_title(title)
  plt.ylabel('Gold Labels')
  plt.xlabel('Predicted Labels')
  plt.show()

def get_report(gold_labels, predicted_labels):
  """ Get classification report for a model
  
  :param gold_labels: actual labels of the dataset
  :type gold_labels: numpy array / python list
  :param predicted_labels: the predictions from the model
  :type predicted_labels: numpy array / python list
  :returns: accuracy, f1 score, precision, recall of the predictions
  :rtype: float
  :returns: confusion matrix for the predictions
  :rtype: python list """

  report_dict = classification_report(gold_labels, predicted_labels, output_dict=True, zero_division=0)
  f1_score = report_dict['macro avg']['f1-score']
  precision = report_dict['macro avg']['precision']
  recall = report_dict['macro avg']['recall']
  accuracy = accuracy_score(gold_labels, predicted_labels)
  confmat = confusion_matrix(gold_labels, predicted_labels)
  
  return accuracy, f1_score, precision, recall, confmat

"""#### 2.2 kNN Classification

Performing Cross Validation
"""

def skl_runKNN():
  """ Perform 5-fold cross validation on the embedded and bag of words tweets """

  skl_conf_matrices = []
  skl_classification_acc_b=[]
  skl_classification_acc_e=[]
  skl_f1_score_b=[]
  skl_f1_score_e=[]

  # running for values of k from 1 to 10
  for i in range(1,11):

    # 5-fold cross validation for embedded tweets using Euclidean distance
    knnclassifier_e = KNeighborsClassifier(n_neighbors=i,p=2,weights='distance')
    knnclassifier_e.fit(training_embeddings,gold_labels_training)
    predicted_labels_e = cross_val_predict(knnclassifier_e, training_embeddings,gold_labels_training, cv=5)
    accuracy_e, f1_score_e, precision_e, recall_e, confmat_e = get_report(gold_labels_training, predicted_labels_e)

    # 5-fold cross validation for bag of words tweets using Euclidean distance
    knnclassifier_b = KNeighborsClassifier(n_neighbors=i,p=2,weights='distance')
    knnclassifier_b.fit(bow_np,gold_labels_training)
    predicted_labels_b = cross_val_predict(knnclassifier_b, bow_np,gold_labels_training, cv=5)
    accuracy_b, f1_score_b, precision_b, recall_b, confmat_b = get_report(gold_labels_training, predicted_labels_b)

    # storing accuracy values and f1 scores
    skl_conf_matrices.append([confmat_e,confmat_b])
    skl_classification_acc_e.append(accuracy_e)
    skl_classification_acc_b.append(accuracy_b)
    skl_f1_score_e.append(f1_score_e)
    skl_f1_score_b.append(f1_score_b)

  # values of k
  vals_k = [i for i in range(1,11)] 
  
  # reporting best values of k for embedded and bag of words tweets and plotting the accuracy and f1 scores
  best_k_f1_e, best_k_acc_e = skl_plotting_function_e_vs_m(vals_k,skl_classification_acc_e,skl_classification_acc_b,skl_f1_score_e,skl_f1_score_b)

# perform 5-fold cross validation on embedded and bag of words tweets
skl_runKNN()

"""Testing"""

def skl_knn_final_stretch(best_k_e, best_k_b):
  """ Evaluate kNN classifier on test tweets using best k values from cross validation
  
  :param best_k_e: best k value for embedded tweets
  :type best_k_e: int
  :param best_k_b: best k value for bag of words tweets
  :type best_k_b: int """

  # fitting both classifiers
  knnclassifier_e = KNeighborsClassifier(n_neighbors=best_k_e, p=2, weights='distance')
  knnclassifier_b = KNeighborsClassifier(n_neighbors=best_k_b, p=2, weights='distance')
  knnclassifier_e.fit(training_embeddings, gold_labels_training)
  knnclassifier_b.fit(bow_np, gold_labels_training)

  # making predictions
  predicted_labels_e = knnclassifier_e.predict(test_embeddings)
  predicted_labels_b = knnclassifier_b.predict(bow_np_test)

  # getting classification report
  accuracy_e, f1_score_e, precision_e, recall_e, confmat_e = get_report(gold_labels_test, predicted_labels_e)
  accuracy_b, f1_score_b, precision_b, recall_b, confmat_b = get_report(gold_labels_test, predicted_labels_b)

  print("Embeddings: \t\t\tBag of Words:")
  print(f"Accuracy: {np.round(accuracy_e*100,4)}% \t\tAccuracy: {np.round(accuracy_b*100,4)}%")
  print(f"F1-Score: {np.round(f1_score_e,4)} \t\tF1-Score: {np.round(f1_score_b,4)}")
  print(f"Precision: {np.round(precision_e,4)} \t\tPrecision: {np.round(precision_b,4)}")
  print(f"Recall: {np.round(recall_e,4)} \t\t\tRecall: {np.round(recall_b,4)}\n")

  # plotting confusion matrices
  title = "Confusion matrix using Embeddings for K="+str(best_k_e)
  labels = ['espn','Google','ndtv','Starbucks','VP','urstrulyMahesh']
  plot_confusion_matrix(confmat_e, title, labels)
  title = "Confusion matrix using Bag of Words for K="+str(best_k_b)
  plot_confusion_matrix(confmat_b, title, labels)

# evaluating kNN classifer on test datasets
skl_knn_final_stretch(4,2)

"""## 3.0 Neural Networks"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

"""#### 3.1 Generating Classifier"""

def nn_classifier(training_sample, gold_labels_training, testing_sample, gold_labels_testing, layer_sizes, title, cross_val=False):
  """ Evaluate Neural Network Classifer on the given datasets
  
  :param training sample: the training tweets
  :type training sample: numpy array / python list
  :param gold_labels_training: actual labels for training tweets
  :type gold_labels_training: numpy array / python list
  :param testing_sample: the test tweets
  :type testing_sample: numpy array / python list
  :param gold_labels_testing: actual labels for the test tweets
  :type gold_labels_testing: numpy array / python list
  :param layer_sizes: shapes of the hidden layers
  :type layer_sizes: tuple
  :param title: title for the confusion matrix
  :type title: string
  :param cross_val: True if cross validation is to be performed, False if normal classification is to be performed
  :type cross_val: bool """

  mlp_classifier = MLPClassifier(random_state=15,hidden_layer_sizes=layer_sizes, batch_size=32, max_iter=300, activation='relu')
  mlp_classifier.fit(training_sample, gold_labels_training)
  if cross_val:
    predicted = cross_val_predict(mlp_classifier, training_sample, gold_labels_training, cv=5)
    accuracy, f1_score, precision, recall, confmat = get_report(gold_labels_training, predicted)
  else:
    predicted = mlp_classifier.predict(testing_sample)
    accuracy, f1_score, precision, recall, confmat = get_report(gold_labels_testing, predicted)

  print(f"Accuracy: {np.round(accuracy*100,4)}%")
  print(f"F1 score: {np.round(f1_score,4)}")
  print(f"Precision: {np.round(precision,4)}")
  print(f"Recall: {np.round(recall,4)}")
  print()
  labels = ['espn','Google','ndtv','Starbucks','VP','urstrulyMahesh']
  plot_confusion_matrix(confmat, title, labels)

"""#### 3.2 Performing k-Fold Cross Validation"""

# 5-Fold cross validation on embedded tweets
print("5-Fold Cross Validation on Neural Network Model on Embedded Features...\n")
title = "Confusion Matrix using Embeddings for Neural Network"
nn_classifier(training_embeddings, gold_labels_training, test_embeddings, gold_labels_test, (256,256,64), title, cross_val=True)

# 5-Fold cross validation on bag of words tweets
print("5-Fold Cross Validation on Neural Network Model on Bag of Words Features...\n")
title = "Confusion Matrix using Bag of Words for Neural Network"
nn_classifier(bow_np, gold_labels_training, bow_np_test, gold_labels_test, (400,256,64), title, cross_val=True)

"""#### 3.3 Testing on Embedded Features"""

print("Testing Neural Network Model on Embedded Features...\n")
title = "Confusion Matrix using Embeddings for Neural Network"
nn_classifier(training_embeddings, gold_labels_training, test_embeddings, gold_labels_test, (256,256,64), title)

"""#### 3.4 Testing on Bag of Words Features"""

print("Testing Neural Network Model on Bag of Words Features...\n")
title = "Confusion Matrix using Bag of Words for Neural Network"
nn_classifier(bow_np, gold_labels_training, bow_np_test, gold_labels_test, (400,256,64), title)

"""## 4.0 Ensemble Methods"""

def ensemble_classifier(classifier, training_sample, gold_labels_training, testing_sample, gold_labels_testing, title):
  """ Evaluate Ensemble Classifer on the given datasets
  
  :param classifier: the classifier to fit and predict on
  :type classifier: scikit-learn classifier
  :param training sample: the training tweets
  :type training sample: numpy array / python list
  :param gold_labels_training: actual labels for training tweets
  :type gold_labels_training: numpy array / python list
  :param testing_sample: the test tweets
  :type testing_sample: numpy array / python list
  :param gold_labels_testing: actual labels for the test tweets
  :type gold_labels_testing: numpy array / python list
  :param layer_sizes: shapes of the hidden layers
  :type layer_sizes: tuple
  :param title: title for the confusion matrix
  :type title: string """

  # ensemble_classifier = BaggingClassifier(base_estimator=SVC(), random_state=1, verbose=verbose, n_jobs=-1)
  classifier.fit(training_sample, gold_labels_training)
  predicted = classifier.predict(testing_sample)
  accuracy, f1_score, precision, recall, confmat = get_report(gold_labels_testing, predicted)

  print(f"Accuracy: {np.round(accuracy*100,4)}%")
  print(f"F1 score: {np.round(f1_score,4)}")
  print(f"Precision: {np.round(precision,4)}")
  print(f"Recall: {np.round(recall,4)}")
  print()
  labels = ['espn','Google','ndtv','Starbucks','VP','urstrulyMahesh']
  plot_confusion_matrix(confmat, title, labels)

"""#### 4.1 Bagging Classifier"""

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

bagging_classifier = BaggingClassifier(base_estimator=SVC(), random_state=15, verbose=False, n_jobs=-1)

# Testing bagging on embedded tweets
print("Applying Bagging Ensemble Method on Embedded Features...\n")
title = "Confusion Matrix using Embeddings on Bagging Classifier"
ensemble_classifier(bagging_classifier, training_embeddings, gold_labels_training, test_embeddings, gold_labels_test, title)

# testing bagging on bag of words tweets
print("Applying Bagging Ensemble Method on Bag of Words Features...\n")
title = "Confusion Matrix using Bag of Words on Bagging Classifier"
ensemble_classifier(bagging_classifier, bow_np, gold_labels_training, bow_np_test, gold_labels_test, title)

"""#### 4.2 Gradient Boosting Classifier"""

from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=15)

# testing gradient boosting on embedded tweets
print("Applying Gradient Boosting Ensemble Method on Embedded Features...\n")
title = "Confusion Matrix using Embeddings on Gradient Boosting Classifier"
ensemble_classifier(gradient_boosting_classifier, training_embeddings, gold_labels_training, test_embeddings, gold_labels_test, title)

# testing gradient boosting on bag of words tweets
print("Applying Gradient Boosting Ensemble Method on Bag of Words Features...\n")
title = "Confusion Matrix using Bag of Words on Gradient Boosting Classifier"
ensemble_classifier(gradient_boosting_classifier, bow_np, gold_labels_training, bow_np_test, gold_labels_test, title)

"""#### 4.3 Voting Classifier"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

clf1 = LogisticRegression(multi_class='multinomial', random_state=20, max_iter=500)
clf2 = RandomForestClassifier(random_state=15)
voting_classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft')

# testing voting on embedded tweets
print("Applying Voting Ensemble Method on Embedded Features...\n")
title = "Confusion Matrix using Embeddings on Voting Classifier"
ensemble_classifier(voting_classifier, training_embeddings, gold_labels_training, test_embeddings, gold_labels_test, title)

# testing voting on bag of words tweets
print("Applying Voting Ensemble Method on Bag of Words Features...\n")
title = "Confusion Matrix using Bag of Words on Voting Classifier"
ensemble_classifier(voting_classifier, bow_np, gold_labels_training, bow_np_test, gold_labels_test, title)

"""#### 4.4 Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(max_features='sqrt', n_estimators=500, random_state=15, n_jobs=-1)

# testing random forest on embedded tweets
print("Applying Random Forest Ensemble Method on Embedded Features...\n")
title = "Confusion Matrix using Embeddings on Random Forest Classifier"
ensemble_classifier(random_forest_classifier, training_embeddings, gold_labels_training, test_embeddings, gold_labels_test, title)

# testing random forest on bag of words tweets
print("Applying Random Forest Ensemble Method on Bag of Words Features...\n")
title = "Confusion Matrix using Bag of Words on Random Forest Classifier"
ensemble_classifier(random_forest_classifier, bow_np, gold_labels_training, bow_np_test, gold_labels_test, title)

"""#### 4.5 Ada-Boost Classifer"""

from sklearn.ensemble import AdaBoostClassifier

adaboost_classifier = AdaBoostClassifier(n_estimators=500, learning_rate=0.1, random_state=15)

# testing ada-boost forest on embedded tweets
print("Applying Ada-Boost Ensemble Method on Embedded Features...\n")
title = "Confusion Matrix using Embeddings on Ada-Boost Classifier"
ensemble_classifier(adaboost_classifier, training_embeddings, gold_labels_training, test_embeddings, gold_labels_test, title)

# testing random forest on bag of words tweets
print("Applying Ada-Boost Ensemble Method on Bag of Words Features...\n")
title = "Confusion Matrix using Bag of Words on Ada-Boost Classifier"
ensemble_classifier(adaboost_classifier, bow_np, gold_labels_training, bow_np_test, gold_labels_test, title)

"""\\

## 5.0 Theoretical Understanding

\\

#### 5.1 Which model performed best and why do you think that is?


With Ensemble methods, the bagging classifier using a support vector machine as the base estimator performed the best with an accuracy of 95.0% on the embedded features.  For the bag of word features, the voting classifier (used for classification) using Logistic regression and the random forest classifier performed the best with an accuracy of 91.17% because multiple decision trees were built to fit each training set. Ensemble methods give us better performance regardless of the model, as they have high predictive accuracy because they create multiple models and then are combined to produce improved results, maintaining the model’s generalization and reducing the model error. 

Without the Ensemble methods, the kNN model (with k=4) on embedded features performed the best, achieving an accuracy of 95.08%. For the bag of word features, the Neural Network Model achieved the best accuracy of 76.5%. They recognize the hidden patterns and identify significant features and correlations in the data. They have the ability to learn and model non-linear and complex relationships. 

\\

#### 5.2 Which features gave better results for each model? Explain.

For the KNN model, embeddings achieved better accuracy of 95.08% as compared to the bag of words which achieved an accuracy of 69.83%. For Neural Networks, embedded features gave a better accuracy of 93.33% as compared to the bag of word features, which gave an accuracy of 76.5%. According to these results, the embedded features achieved the best accuracy regardless of the model because the word embeddings represent word meanings based on their occurrence in a text corpus. When the text corpus is very large (in our case, which was approximately 6000), the word embedding vectors represent the general meaning of the word, as it would be of great value in several applications. Word Embeddings allow us to extract features from them to learn the stylistic patterns of authors based on context and co-occurrence of the words in the field of Authorship Attribution.

\\

#### 5.3 What effect would increasing the classes to 150 have?

Increasing the number of classes would lower the accuracy of the machine-learning models because we are increasing the number of unique authors that we have to distinguish between. If we have more authors, the similarity of tweets between 2 different authors might increase, so we would have to look for more specific and subtle differences in the vocabulary, style and writing patterns which would be a more difficult task.

\\

#### 5.4 Suggest improvements to text preparation, feature extraction, and models that can be made to perform this task better.

Text Preparation:

1. We could have applied the lemmatization technique to normalize the text as it would have taken into consideration the context of the word to determine the intended meaning we were looking for. It would have removed the inflections and mapped the word to its root form.

2. Text enrichment/augmentation could have been implemented as well, as it provides more semantics to our original text, improving the predictive power and the depth of analysis we were to perform on the data. We could have used part-of-speech tagging to get more granular information about the words in our text. 

Feature Extraction:

1. It could have been improved through filter methods as they work by ranking the features and removing the features falling within a certain threshold. Low-variance features (constant features) could have been removed as they do not provide significant information. Moreover, features could have been selected by assessing the strength of the relationship with the target and by removing the highly correlated features by multicollinearity. For example, through correlation, chi-squared test and ANOVA.

2. Wrapper methods could be applied, which include forward selection, backward selection, and recursive feature elimination. Wrapper methods evaluate the specific machine learning algorithm to find optimal features. Forward selection methods start with the best single feature and progressively add the best-performing remaining features, whereas the backward selection method starts with all the features. Lastly, we could select features through the recursive feature elimination method, which removes the least important features and recursively trains the models with the remaining features, repeating the whole process until a desired number of features is reached. However, this has high computation time and cost. 

Models:

  KNN: 

  1. By reducing the dimensionality of the dataset

  2. Using the grid-search method to find optimal parameters

Neural Networks

  1. Increasing the hidden layers

  2. We could have experimented with the different activation functions in the intermediate and the output layers to see which gives the best results

  3. Increase the number of neurons per layer; ideally, they should be twice the size of the input layers 

SVMs

  1. Experimenting with different kernels and checking which improves the    performance

  2. Optimizing the parameters by hyperparameter tuning

Random Forest (most powerful ensemble classifier)

  1. Reducing the depth in the case of overfitting

  2. Tuning the hyperparameters of the model using GridSearchCV (tweaking the number of trees or max nodes and checking the effect on accuracy)

  3. Extra forest (ensemble supervised machine learning algorithm) could have been implemented, as it would increase the accuracy, is much faster and causes a reduction in bias 

Logistic Regression

  1. Optimizing log loss and F1 score

  2. Hyperparameter tuning through Grid Search as it would improve accuracy


\\

#### 5.5 What -- in your understanding -- are the applications of authorship attribution?

Authorship attribution, known as author recognition or author verification,helps us identify the author of the tweet or document. It is used in the literature and is used to define the characterisation of documents that capture the writing style of the authors. It can be used to discover the author of an ancient or religious manuscript to gather the texts by the relevant authors. Part-Of-Speech n-grams and frequencies of function words are reliable and effective for authorship attribution. It is used for document classification and makes use of stylometry methods. It is also used in the identification of cybercrimes and in fraud detection. Moreover, Intrinsic Plagiarism Detection could also be done and categorized through the authorship attribution process. Authorship Attribution can be used in code authorship, which can be used in plagiarism detection through the identification of writing styles of the contributions made by the authors. Furthermore, this technology can also be used to solve copyright disputes in cases where there is a disagreement on who wrote a particular piece of work. For instance, gender identification could be made through author attribution. The gender of a text documents author could be predicted along with their location and ethnicity (black, brown, white), religion to some extent, mental state (psychologically well or ill) and, for instance, the accent of English which would be different for Pakistanis and Americans in terms of the text written by each could also be determined through the text. It is also used in cybersecurity as it identifies the authors (deceptive authors) across the social media who misuse the identity of the original author as their own. 


\\
"""