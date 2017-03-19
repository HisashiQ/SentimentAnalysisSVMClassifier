# coding: utf-8

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
from nltk.stem import WordNetLemmatizer
# DATA LOADING AND PARSING

# convert line from input file into a datetime/string pair
def parseTweet(tweetLine):
    timestamp = datetime.strptime(tweetLine[1], "%Y-%m-%d %H:%M:%S")
    content = tweetLine[4]
    return (timestamp, content)

# load data from a file and append it to the tweetData
def loadData(path, label, tweet=None):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        next(reader)
        for line in reader:
            (dt,tweet) = parseTweet(line)
            tweetData.append((dt,preProcess(tweet),label))
            trainData.append((toFeatureVector(preProcess(tweet)),label))



# load application data
def loadApplicationData(path):
    with open(path, 'rb') as f:
        reader = unicodecsv.reader(f, encoding='utf-8')
        next(reader)
        for line in reader:
            (dt,data) = parseTweet(line)
            if dt.day == 9:
                londonTweetData.append((dt,data))
        return londonTweetData
# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# input: a string of one tweet
def preProcess(text):
    # lemmatizer = WordNetLemmatizer()
    token = TweetTokenizer()
    #Replace each URL with 'REPLACEurl'
    # noUrl = re.sub(r'(?:(http://)|(www\.))(\S+\b/?)([!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]*)(\s|$)', ' REPLACEDurl ', text)
    # noUsers = re.sub(r'@([a-z0-9_]+)', '', noUrl)
    tokenList = token.tokenize(text)
    # tokenList = []
    # for i in t:
    #     tokenList.append(lemmatizer.lemmatize(i))
    # return [word for word in tokenList if word not in stopwords.words('english')]
    return tokenList

# input: a tokenised sequence
# you can optionally keep track of a global feature list that is a list of all the features (or words) that you encounter while going through the dataset
featureDict = {}
def toFeatureVector(words):
    tweetHash = {}
    for word in words:
        if word in tweetHash:
            # if word == "😄":
                # tweetHash[word] += 10
            # elif word == "#angry":
            #     tweetHash[word] += 10
            # else:
            #     tweetHash[word] += 1
            tweetHash[word] += 1
        else:
            tweetHash[word] = 1
    for w in words:
        if w in featureDict:
            pass
        else:
            featureDict[w] = 1
    return tweetHash

# TRAINING AND VALIDATING OUR CLASSIFIER

def trainClassifier(trainData):
    print("Training Classifier...")
    return SklearnClassifier(LinearSVC()).train(trainData)

def crossValidate(dataset, folds):
    shuffle(dataset)
    foldSize = len(dataset)//folds
    prec = 0
    rec = 0
    f = 0
    accuracy = 0
    results = []
    for i in range(folds):
        realLabels = []

        testData = dataset[i*foldSize:][:foldSize]
        trainingData = dataset[:i*foldSize] + dataset[(i+1)*foldSize:]

        for j in testData:
            (msg, label) = j
            realLabels.append(label)

        classifier = trainClassifier(trainingData)
        totalValues = precision_recall_fscore_support(realLabels, predictLabels(testData,  classifier), average='macro')
        prec += totalValues[0]
        rec += totalValues[1]
        f += totalValues[2]
        accuracy += accuracy_score(realLabels, predictLabels(testData,  classifier))

    print(prec)
    print(rec)
    print(f)
    print(accuracy)
    results.append((prec/folds, rec/(folds), f/folds, accuracy/folds))
    return results

# PREDICTING LABELS GIVEN A CLASSIFIER
#returns label
def predictLabels(tweetData, classifier):
	return classifier.classify_many(map(lambda t: t[0], tweetData))

def predictLabel(text, classifier):
	return classifier.classify(toFeatureVector(preProcess(text)))

# COMPUTING ANGER LEVEL ON A SET OF TWEETS

def findAngerLevels(tweetData, classifier):
    cont = True
    currentTime = 23
    angryNum = 0
    angry = []
    totalV = 0
    for single in tweetData:
        (dt, msg) = single
        if dt.hour == currentTime:
            totalV += 1
            if predictLabel(msg, classifier) == "angry":
                angryNum += 1
        else:
            angry.append((currentTime, (angryNum/totalV), angryNum))
            currentTime -= 1
            totalV = 0
            angryNum = 0

    totalV = 0
    angryNum = 0

    for j in tweetData:
        (dt, msg) = j
        if dt.hour == 0:
            totalV += 1
            if predictLabel(msg, classifier) == "angry":
                angryNum += 1

    angry.append((0, (angryNum/totalV), angryNum))

    return angry

tweetData = []
trainData = []
londonTweetData = []

# the output classes
angryLabel = 'angry'
happyLabel = 'happy'

# references to the data files
angryPath = 'angry_tweets.csv'
happyPath = 'happy_tweets.csv'
londonPath = 'london_2017_tweets.csv'

# In order to test the code in this
#
# do the actual stuff
print("Loading happy tweets...")
loadData(happyPath, happyLabel)
print("Loading angry tweets...")
loadData(angryPath, angryLabel)
cv_results = crossValidate(trainData, 10)
print('number of words: ' + str(len(featureDict)))
print("Precision Average: {}\nRecall Average: {}\nF Score Average: {}\nAccuracy Average: {}".format(cv_results[0][0], cv_results[0][1], cv_results[0][2], cv_results[0][3]))
classifier = trainClassifier(trainData)
print("Loading London data")
loadApplicationData(londonPath)
print("Computing anger levels!")
angerLevels = findAngerLevels(londonTweetData, classifier)
anger_peaks = sorted(angerLevels, key=lambda t: t[1], reverse=True)
for i in anger_peaks[:10]:
    print("Hour: {}".format(i[0]))
    print("   % angry tweets vs total tweets in hour: {}".format(i[1]))
    print("   # of angry tweets: {}".format(i[2]))
