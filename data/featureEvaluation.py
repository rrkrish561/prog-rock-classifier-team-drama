import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def assess_features(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=11)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(y_pred)
    print('Accuracy:', accuracy_score(y_test, y_pred))

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(y_pred)
    print('Accuracy:', accuracy_score(y_test, y_pred))

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    feature_importances = clf.feature_importances_

    print('Feature Importance based on Decision Tree')
    for idx, val in enumerate(feature_importances):
        print(FEATURE_NAMES[idx], val)
    plt.bar(list(range(len(feature_importances))), feature_importances)
    plt.xticks(list(range(len(feature_importances))), FEATURE_NAMES)
    plt.title('Feature Importance based on Decision Tree')
    plt.show()
    print()

    clf = RandomForestClassifier()
    clf.fit(X, y)
    feature_importances = clf.feature_importances_

    print('Feature Importance based on Random Forest')
    for idx, val in enumerate(feature_importances):
        print(FEATURE_NAMES[idx], val)
    plt.bar(list(range(len(feature_importances))), feature_importances)
    plt.xticks(list(range(len(feature_importances))), FEATURE_NAMES)
    plt.title('Feature Importance based on Random Forest')
    plt.show()


if __name__ == '__main__':

    X, y = None, None

    with open('X.json', 'r') as X_file:
        X = json.load(X_file)
    with open('y.json', 'r') as y_file:
        y = json.load(y_file)

    feature_names = ['Duration', 'Mean Amplitude', 'Mean RMSE', 'ZCR', 'Mean Spectral Centroid', 'Mean Spectral Rolloff RP=0.1', 'Mean Spectral Rolloff RP=0.2', 'Mean Spectral Rolloff RP=0.3', 'Mean Spectral Rolloff RP=0.4', 'Mean Spectral Rolloff RP=0.5', 'Mean Spectral Rolloff RP=0.6', 'Mean Spectral Rolloff RP=0.7', 'Mean Spectral Rolloff RP=0.8', 'Mean Spectral Rolloff RP=0.9', 'Mean MFCC 1', 'Mean MFCC 2', 'Mean MFCC 3', 'Mean MFCC 4',
                     'Mean MFCC 5', 'Mean MFCC 6', 'Mean MFCC 7', 'Mean MFCC 8', 'Mean MFCC 9', 'Mean MFCC 10', 'Mean MFCC 11', 'Mean MFCC 12', 'Mean MFCC 13', 'Mean MFCC 14', 'Mean MFCC 15', 'Mean MFCC 16', 'Mean MFCC 17', 'Mean MFCC 18', 'Mean MFCC 19', 'Mean MFCC 20', 'Mean Spectral Contrast 1', 'Mean Spectral Contrast 2', 'Mean Spectral Contrast 3', 'Mean Spectral Contrast 4', 'Mean Spectral Contrast 5', 'Mean Spectral Contrast 6', 'Mean Spectral Contrast 7']

    assess_features(X, y, feature_names)
