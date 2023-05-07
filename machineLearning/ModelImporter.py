import pandas as pd
import numpy as np
import time
import pickle
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
''' Run this file to make a new Model from the MasterPoseData.xlsx data'''



def ImportModel(fileName, printName=False):
    '''A function that deserializes a machine learning model from pickle.

    Parameters:
    
    fileName - string
    The name of the file to be imported. Must end in ".pickle".
    
    printName - bool
    Whether or not you want it to print the name of the model, to indicate which it is.'''

    import pickle

    # Deserialization
    with open(fileName, "rb") as infile:
        clf = pickle.load(infile)
    
    if printName == True:
        print(clf)
     
    return clf

def makeModel():
    df = pd.read_excel('MasterPoseData\MasterPoseData.xlsx')

    # Getting the feature names
    all_col_names = df.columns.tolist()
    feature_names = all_col_names[:len(all_col_names)-1]

        
    # Setting up X (features & data; must be a numpy array)
    X = df.iloc[:,:len(feature_names)].to_numpy()


    # Setting up Y (classifications/results; can be a simple list)
    y = df.iloc[:,len(feature_names)].tolist()



    # Splitting data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

    
    # A few possible models to use
    models = (
    # SVC with linear kernel [index 0]
    svm.SVC(kernel="linear", C=1.0),
        
    # SVC with RBF kernel [index 1]
    svm.SVC(kernel="rbf", gamma=0.7, C=1.0),
        
    # k-Nearest Neighbors [index 2]
    KNeighborsClassifier(),
    )



    clfs = []

    for i in range(len(models)):
    
        clf = models[i]

        clfs.append(clf)

        clf.fit(X_train, Y_train)

        print("Model =",clf)

        # Testing the Model
        score = round(clf.score(X_test,Y_test),3)
        print("score:",str(score))

        # Measuring Prediction Time
        tic = time.perf_counter()
        clf.predict(X_test[0:100])
        toc = time.perf_counter()
        print("prediction time:",str(round(toc-tic,3))+" seconds\n")

    # Entering your choice
    print("Model Choices")
    print("-------------")
    print("SVC(kernel='linear') : 0")
    print("SVC(gamma=0.7) : 1")
    print("KNeighborsClassifier() : 2")
    while True:
        try:
            ModelChoice = int(input("\nEnter your choice (0, 1, or 2):"))
            if ModelChoice == 0 or ModelChoice == 1 or ModelChoice == 2:
                break
        except ValueError:
            print("Please enter a valid number.")
        else:
            print("Please enter 0, 1, or 2")

    ModelChoice = 0

    # Choosing the model
    clf = clfs[ModelChoice]

    print("You chose the model:", clf)

    # Serialization
    with open("MLmodel.pickle", "wb") as outfile:
        pickle.dump(clf, outfile)


def main():
    makeModel()

if __name__ == "__main__":
    main()