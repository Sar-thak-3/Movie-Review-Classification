# MULTINOMIAL EVENT MODEL
x = ["This is awesome movie",
     "Great movie! i liked it alot",
     "Happy ending awesome acting by the hero",
     "Loved it! truly great",
     "bad not upto the mark",
     "could have been better",
     "surely a dissapointing movie"]

y = [1,1,1,1,0,0,0] # positive -> 1 ,negative-> 0
x_test = ["I was happy & happy and i loved the acting in the movie",
        "the movie i saw was bad"]

import workingWithFiles as ct
x_clean = [ct.getCleanReview(i) for i in x]  # list comprehension
xt_clean = [ct.getCleanReview(i) for i in x_test]

print(xt_clean)

# VECTORIZATION
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2))  # includes both unigram and bigram
x_vec = cv.fit_transform(x_clean).toarray()
print(x_vec)  # make vector for trainig data according to the position of words in vocab
print(x_vec.shape)
print(cv.get_feature_names())  # gives vocab

# Vectorisation on test data
# xt_vec = cv.fit_transform(xt_clean).toarray()   # cv.fit_transform  fit the words in vocab as well as transfrom them to dictionary among them
# print(xt_vec)
# but we do not require this we want to transform the words of test into train vocab
xt_vec = cv.transform(xt_clean).toarray()
print(xt_vec)
print(xt_vec.shape)


# Multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
mnb = MultinomialNB()
print(mnb)

# TRAINING
mnb.fit(x_vec,y)
yt_pred = mnb.predict(xt_vec)
y_pred = mnb.predict(x_vec)
y_pred = y_pred.reshape((-1,))
print(y_pred)
print("multinomial",mnb.predict_proba(xt_vec)) # probabilities of x_test belongs to class 1 or 2


# MULTIVARIATE BERNOULLI EVENT MODEL
bnb = BernoulliNB(binarize=0.0)
bnb.fit(x_vec,y)
print("binomial",bnb.predict_proba(xt_vec))
print(bnb.predict(xt_vec))
print("bnb score",bnb.score(x_vec,y)*100)


# GENERATE CONFUSIPON MATRIX
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
import matplotlib.pyplot as plt
cnf_matrix = confusion_matrix(y,y_pred)
print(cnf_matrix)

# plot_confusion_matrix(cnf_matrix,normalize=False,cmap=plt.cm.Blues)