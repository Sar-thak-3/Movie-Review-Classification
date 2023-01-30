from nltk.tokenize import RegexpTokenizer  # custom tokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import sys

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getStemmerReview(review):
    print(review)
    review = review.lower()
    review = review.replace("<br ></br >"," ")

    # tokenization
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    cleaned_review = ' '.join(stemmed_tokens)
    return cleaned_review

def getCleanDocument(inputFile,outputFile):
    out = open(outputFile,'w',encoding='utf8')

    with open(inputFile,encoding='utf8') as f:
        reviews = f.readlines()

    for review in reviews:
        # print(review)
        cleaned_review = getStemmerReview(review)
        print((cleaned_review),file=out)
    out.close()

inputFile = 'toy.txt'
outputFile = 'output.txt'
getCleanDocument(inputFile,outputFile)


def getCleanReview(review):
    review = review.lower()
    review = review.replace("<br ></br >"," ")

    # tokenization
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    cleaned_review = ' '.join(stemmed_tokens)
    return cleaned_review