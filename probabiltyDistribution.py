
# coding: utf-8

from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, LaplaceProbDist, MLEProbDist
from nltk.util import bigrams
import unicodecsv
import nltk
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from nltk.tokenize import TweetTokenizer
from random import shuffle
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import re
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
### PART B ###
##############
####
## A bigram model using the NLTK built-in functions
####

# given a list of lists of preprocessed tweets,
# getBigrams should return a list of pairs containing all the bigrams that
# are observed in the list.

def getBigrams(tweets):
    a = []
    for i in tweets:
        a.append(bigrams(i))
    bigramsArr = [item for sublist in a for item in sublist]
    return bigramsArr

# conditionalProbDist will return a probability distribution over a list of
# bigrams, together with a specified probability distribution constructor
def conditionalProbDist(probDist, bigrams):
    cfDist = ConditionalFreqDist(bigrams)
    cpDist = ConditionalProbDist(cfDist, probDist, bins=len(bigrams))
    return cpDist

londonTweetData = []
def loadData(path):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        next(reader)
        for line in reader:
            (dt, tweet) = parseTweet(line)
            londonTweetData.append((dt, (preProcess(tweet))) )

def preProcess(text):
    token = TweetTokenizer()
    # lowercase = text.lower()
    # changeHashTag = re.sub("#tubestrike", "tube strike", lowercase)
    tokenList = token.tokenize(text)
    return tokenList

def parseTweet(tweetLine):
    timestamp = datetime.strptime(tweetLine[1], "%Y-%m-%d %H:%M:%S")
    content = tweetLine[4]
    return (timestamp, content)

# this is the function where you can put your main script, which you can then
# toggle if for test purposes
def mainScript():
    loadData("london_2017_tweets.csv")
    #Uncomment below line and comment out 'factory = MLEProbDist' to use LaplaceProbDist as the factory
    # factory = LaplaceProbDist
    factory = MLEProbDist
    wholeDatasetProb = calculateProp(getContentWholeSet(), factory)
    fifthDatasetProb = calculateProp(getContentFifth(), factory)
    ninthDataSetProb = calculateProp(getContentNinth(), factory)
    print("Whole dataset: {}".format((wholeDatasetProb["tube"].prob("strike"))))
    print("5th of Jan: {}".format((fifthDatasetProb["tube"].prob("strike"))))
    print("9th of Jan: {}".format((ninthDataSetProb["tube"].prob("strike"))))

    fifth = getRatio(fifthDatasetProb, wholeDatasetProb)
    ninth = getRatio(ninthDataSetProb, wholeDatasetProb)

    sortedFifth = sorted(fifth, key=lambda t: t[1], reverse=True)
    print("Fifth vs Whole month:   \n{}".format(sortedFifth[:10]))
    sortedNinth = sorted(ninth, key=lambda t: t[1], reverse=True)
    print("Ninth vs Whole month:   \n{}".format(sortedNinth[:10]))

def getRatio(specificDayProb, wholeDatasetProb):
    dayArr = []
    for i in specificDayProb:
        for w in specificDayProb[i].samples():
            wholeProb = wholeDatasetProb[i].prob(w)
            dayArr.append([(i,w), (specificDayProb[i].prob(w) - wholeProb), specificDayProb[i].prob(w)])
    return dayArr

def calculateProp(bigrams, factory):
    bi = getBigrams(bigrams)
    probD = conditionalProbDist(factory, bi)
    return probD

def getContentWholeSet():
    content = []
    for singleTweet in londonTweetData:
        (timestamp, msg) = singleTweet
        content.append(msg)
    return content

def getContentFifth():
    cont = []
    for singleTweet in londonTweetData:
        (timestamp, msg) = singleTweet
        if timestamp.day == 5:
            cont.append(msg)
    return cont

def getContentNinth():
    content = []
    for singleTweet in londonTweetData:
        (timestamp, msg) = singleTweet
        if timestamp.day == 9:
            content.append(msg)
    return content

# The line below can be toggled as a comment to toggle execution of the main script
results = mainScript()
