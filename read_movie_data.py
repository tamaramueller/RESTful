import re
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

reviews_train = []
reviews_test = []
reviews_test_clean = []
reviews_train_clean = []


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

def prepare():
    global reviews_test
    global reviews_train 
    global reviews_train_clean
    global reviews_test_clean

    for line in open('aclImdb/movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())
        
    for line in open('aclImdb/movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())

    reviews_train_clean = preprocess_reviews(reviews_train)
    reviews_test_clean = preprocess_reviews(reviews_test)


def sentimentAnalysis(textIn):
    if not os.path.isfile("finalmodel.file"):
        print("Preparing Data")
        prepare()
        print("Training Model")

        cv = CountVectorizer(binary=True)
        cv.fit(reviews_train_clean)
        X = cv.transform(reviews_train_clean)
        X_test = cv.transform(reviews_test_clean)

        target = [1 if i < 12500 else 0 for i in range(25000)]

        X_train, X_val, y_train, y_val = train_test_split(
            X, target, train_size = 0.75
        )

        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            lr = LogisticRegression(C=c)
            lr.fit(X_train, y_train)
            print ("Accuracy for C=%s: %s" 
                % (c, accuracy_score(y_val, lr.predict(X_val))))

        final_model = LogisticRegression(C=0.05)
        final_model.fit(X, target)
        print ("Final Accuracy: %s" 
            % accuracy_score(target, final_model.predict(X_test)))

        with open("finalmodel.file", "wb") as f:
            pickle.dump(final_model, f, pickle.HIGHEST_PROTOCOL)

        with open("cv.file", "wb") as fi:
            pickle.dump(cv, fi, pickle.HIGHEST_PROTOCOL)

        print("file didn't exist")

    else:
        print("file exists")
        with open("finalmodel.file", "rb") as f:
            final_model = pickle.load(f)

        with open("cv.file", "rb") as fi:
            cv = pickle.load(fi)

    emotion = cv.transform([textIn])
    print(final_model.predict(emotion)[0])
    return final_model.predict(emotion)[0]


if __name__ == '__main__':
    sentimentAnalysis("I am happy")