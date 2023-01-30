import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Train.csv')
df = df.values
# print(df.shape,type(df))

x_train = df[:,0]
y_train = df[:,1]

# Natural Language Preprocessing
import nltk

# Tokenisation
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
def tokenization(data):
    x_train_tokenized = [tokenizer.tokenize(sentence) for sentence in data]
    return x_train_tokenized
    # return [['..','..','..'],['..','..','..','..']]  -> 2D array of words from dataFrame x_train with tokenized

# Stopword removal
from nltk.corpus import stopwords
en_stopwords = set(stopwords.words('english'))
def stopwordRemoval(line,en_stopwords):
    new_line = [word for word in line if word not in en_stopwords]
    return new_line
    # return ['..','..','..']  -> 1d array od words which are not in en_stopwords
    
# Stemming
from nltk.stem.snowball import  SnowballStemmer
ss = SnowballStemmer('english')
def stemming(line):
    new_line = []
    for word in line:
        new_line.append(ss.stem(word))
    return new_line
    # return ['..','..','..'] -> 1D array of words which are filtered by snowballStemmer

# Building a vocab
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2))
def vectorization(data):
    matrix = cv.fit_transform(data)
    return matrix


x_train_tokenized = tokenization(x_train[:50])  # --> return [['..','..','..'],['..','..','..','..']]  -> 2D array of words from dataFrame x_train with tokenized

preprocessed = []
for line in x_train_tokenized:
    preprocessed.append(" ".join(stemming(stopwordRemoval(line,en_stopwords))))  # --> return [".................."] ; words in list joined to each other by " "
# preprocessed is a 2D list

matrix = vectorization(preprocessed)  # --> return vectorized building a vocabulary among words in 2D list(preprocessed)
matrix = matrix.toarray()  #--> .toarray() gives 2d array according to the words in dictionary cv.vocabulary_
# print(matrix.shape,type(matrix))
# print(cv.vocabulary_)

# Calculating probability

def prior_probability(y_train,label):
    total_examples = y_train.shape[0]
    class_examples = np.sum(y_train==label)
    return class_examples/float(total_examples)

# print(prior_probability(np.array(['pos','pos','pos','pos','pos','neg','neg','neg','neg','neg','neg']),'pos'))

def conditional_probability(x_train,y_train,label,feature_col,feature_val):
    x_filtered = x_train[y_train==label] # for filtering the x_trains where y_train is equals to the label/class we required
    numerator = np.sum(x_filtered[:,feature_col]==feature_val)
    denominator = x_filtered.shape[0]
    return numerator/float(denominator)

def predict(x_train,y_train,x_test):
    classes = np.unique(y_train)
    n_features = x_train.shape[1]
    post_probs = []
    for label in classes:
        likelihood = 1
        for feature in range(n_features):
            cond = conditional_probability(x_train,y_train,label,feature,x_test[feature])
            likelihood *= cond

        prior_prob = prior_probability(y_train,label)
        post_probs.append(prior_prob*likelihood)
    prediction = np.argmax(post_probs)
    return prediction

# print(type(y_train),y_train.shape)
x_test = pd.read_csv("Test.csv")
x_test = x_test.values
# print(x_test,x_test.shape,type(x_test))
test_data = cv.transform([(" ".join(stemming(stopwordRemoval(tokenization(x_test[0])[0],en_stopwords))))]).toarray()
print(y_train[:5])
print(np.unique(y_train))
print(test_data)
print(predict(matrix,y_train[:50],test_data[0]))