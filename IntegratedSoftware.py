
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_files
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
from scipy.io.arff import loadarff
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import time

# Shuffling Data And Randomly Dividing Whole Dataset
def randomPartition(X, y):
    partition_Ration = 0.4
    number_Rows = X.shape[0]
    number_Train = int(round(number_Rows*partition_Ration))
    number_Test = number_Rows - number_Train
    
    rows = np.arange(number_Rows)
    np.random.shuffle(rows)

    training = rows[:number_Train]
    testing = rows[number_Train:]
    
    y = y.reshape((-1,1))
    
    X_Training = X[training,:]
    y_Training = y[training,:]
        
    y_Training = y_Training.reshape(1, -1)[0]

    return X_Training, y_Training

# Gaussian Naive Bayes
def gaussianNaiveBayes(X, y):
    classifier = GaussianNB()
#     classifier.fit(X, y)
    
    return classifier

# MultiNomial Naive Bayes With Parameters
def multinomialNaiveBayes(X, y, alphaValue, fitPrior, classPrior):
    classifier = MultinomialNB()
    classifier.set_params(alpha = alphaValue, fit_prior = fitPrior, class_prior = classPrior)
#     classifier.fit(X, y)
    
    return classifier

# Bernoulli Naive Bayes With Parameters
def bernoulliNaiveBayes(X, y, alphaValue, binarizeValue, fitPrior, classPrior):
    classifier = BernoulliNB()
    classifier.set_params(alpha = alphaValue, binarize = binarizeValue, fit_prior = fitPrior, class_prior = classPrior)
#     classifier.fit(X, y)
    
    return classifier

# Quadratic Discriminant Analysis
def quadraticDiscriminantAnalysis(X, y, reg_parameter):
    classifier = QuadraticDiscriminantAnalysis()
    
    classifier.set_params(reg_param=reg_parameter)
#     classifier.fit(X, y)
    
    return classifier

# Linear Discriminant Analysis
def linearDiscriminantAnalysis(X, y, solverValue, shrinkageValue):
    classifier = LinearDiscriminantAnalysis(solver = solverValue, shrinkage = shrinkageValue)
#     classifier.set_params(solver = solverValue, shrinkage = shrinkageValue)
#     classifier.fit(X, y)
    
    return classifier

def nearestNeighbor(X, y, numberOfNeighbor, weight):
    classifier = KNeighborsClassifier()
    classifier.set_params(n_neighbors = numberOfNeighbor, weights = weight)
#     classifier.fit(X, y)
    
    return classifier

def SVM_Gaussian(X, y, CValue, gammaValue):
    
    classifier = svm.SVC(C=CValue, kernel='rbf', gamma=gammaValue)
    
#     classifier.fit(X, y)
    
    return classifier

# Reading Data In Format 
def generate_data(dataSet):
    
    if dataSet == "Random":
        X = np.array([[1, 1], [2, 1], [3, 2], [9, 102], [1, 80], [2, 70]])
        y = np.array([2, 2, 2, 1, 2, 9])
        
    elif dataSet == "Letter Recognition":
        
        f = open("Data/letter-recognition.data")
        
        data = np.loadtxt(f, delimiter=',')
#         print data.shape

        X = data[:, 1:]
        y = data[:, 0]
        
        
#         print X
#         print y
    elif dataSet == "InternetAdsDataset":
        f = open("Data/internet-ads.data")
        
        data = np.loadtxt(f, delimiter=',')
#         print data.shape

        y = data[:,-1]
        X = data[:,:-1]
        
    elif dataSet == "UCI_HARS":
        forX = open("Data/UCI_HARS/X_Whole.txt")
        forY = open("Data/UCI_HARS/y_Whole.txt")


        
        dataForX = np.loadtxt(forX)
        dataForY = np.loadtxt(forY)

#         print dataForX.shape

        X = dataForX
        y =dataForY
        
    elif dataSet == "cover-type":
        data = np.loadtxt('Data/covertype.data',delimiter=',')
        y = data[:,-1]
        X = data[:,:-1]
    
    elif dataSet == "farm-ads":
        X, y = load_svmlight_file('Data/farm-ads')
        
        X = X.todense()
        
    elif dataSet == "amazon":
        data=np.genfromtxt('Data/amazon.txt', delimiter = ',')
        y = data[:,-1]
        X = data[:,:-1]
        
#         X = X.todense()
        
    elif dataSet == "adult":
        data=np.loadtxt(fname = 'Data/adult.txt', dtype='str', delimiter = ',')
        enc = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13], sparse = False)

        newData =  enc.fit_transform(data)
        y = newData[:,-1]
        X = newData[:,:-1]
        
    elif dataSet == "DBWorld":
        print "Reading DBWorld Dataset"
        data = loadarff('Data/dbworld_bodies.arff')
#         print data
        
        act_data = []
        for i in range(len(data[0])):
            act_data.append(list(data[0][i]))
# #         # print act_data
        act_data_array = np.asarray(act_data)
#         print "Here"
        partition = act_data_array.shape[1] - 1
        X = act_data_array[:,:partition]
# #         print X.shape
        y = act_data_array[:,partition:].reshape(act_data_array.shape[0])
#         print Y.shape
        
#         X = np.array(X)
#         y = np.array(y)
#         print len(X)
#         dense_vector = np.zeros((X.shape[0], ), int)
#         for inner in X:
#             for index, value in inner:
#                 dense_vector[index] = value
# 
#         y = y.toarray()
#         X = dense_vector
        
    return X, y

# Using Cross Validation, We are choosing best model in respective model criteria. 
def compareAccuracy(X, y, randomFactor, runMulti, performance="accuracy"):
    randomization = randomFactor
    typeOfCompare = performance
    bestAccuracy = 0
    bestModel = ""
    print "-- -- -- --"
    
    print "Running Multinomial? ", runMulti

    
    for i in range(randomization):
        
        print "Iteration", i
        print "-- -- -- --"
#         #     Calculating Gaussian Naive Bayes
# #         print "-- -- -- --"
        
    
        cross_validate=cross_validation.StratifiedKFold(y, 5, shuffle=True);
        gaussianStartTime = time.time()
        print "****Gaussian Naive Bayes****"
        classifierGaussian = gaussianNaiveBayes(X, y)

        print "classifier Gaussian", classifierGaussian

        
# #         print classifierGaussian.predict(X_Testing)
# #         accuracyUsingGaussian = classifierGaussian.score(X_Testing, y_Testing)
        valid_cross = cross_validation.cross_val_score(classifierGaussian, X, y, cv = cross_validate, scoring=typeOfCompare)
# #         print valid_cross
        accuracyUsingGaussian = np.mean(valid_cross)
        print "Accuracy Using Gaussian", accuracyUsingGaussian
        
# #         print "RMSE", valid_cross.mean_squared_error
        if accuracyUsingGaussian == bestAccuracy:
#             bestAccuracy = accuracyUsingGaussian
            bestModel = bestModel + ", Gaussian Naive Bayes"
        if accuracyUsingGaussian > bestAccuracy:
            bestAccuracy = accuracyUsingGaussian
            bestModel = "Gaussian Naive Bayes"
        
        gaussianExecutionTime = time.time() - gaussianStartTime
        print "Execution Time In Gaussian Naive Bayes", str(gaussianExecutionTime)

#         #     Calculating Multinomial Naive Bayes
#         #     X Must Be Non Negative
        print "-- -- -- --"

#         print "-- -- -- --"
        if runMulti:
            multiStartTime = time.time()

            print "****Multinomial Naive Bayes****"

            alphaRange = [0.001, 0.01, 0.1, 1]
            for a in alphaRange:
                print "Alpha ", a
                cross_validate=cross_validation.StratifiedKFold(y, 5, shuffle=True);

                alphaValue = a
                fitPrior=True
                classPrior=None
                classifierMultinomial = multinomialNaiveBayes(X, y, alphaValue, fitPrior, classPrior)

                print "classifier Multinomial", classifierMultinomial


    #             print classifierMultinomial.predict(X_Testing)
    #             print classifierMultinomial.score(X_Testing, y_Testing)

                accuracyUsingMultinomial = np.mean(cross_validation.cross_val_score(classifierMultinomial, X, y, cv = cross_validate, scoring=typeOfCompare))

                print "Accuracy Using Multinomial ", accuracyUsingMultinomial
                
                if accuracyUsingMultinomial == bestAccuracy:
#             bestAccuracy = accuracyUsingGaussian
                    bestModel = bestModel + ", Multinomial Naive Bayes With Alpha " + str(alphaValue)
                if accuracyUsingMultinomial > bestAccuracy:
                    bestAccuracy = accuracyUsingMultinomial 
                    bestModel = "Multinomial Naive Bayes With Alpha " + str(alphaValue)
                    
                
            multiExecutionTime = time.time() - multiStartTime

            print "Multinomial Execution Time", str(multiExecutionTime)

        print "-- -- -- --"

#         print "-- -- -- --"
        bernoulliStartTime = time.time()
        print "****Bernoulli Naive Bayes****"
        alphaRange = [0, 0.001, 0.01, 0.1, 1]
        for a in alphaRange:
            print "Alpha ", a
            cross_validate=cross_validation.StratifiedKFold(y, 5, shuffle=True);

            alphaValue = a
            binarize = 0
            fitPrior=True
            classPrior=None
            classifierBernoulli = bernoulliNaiveBayes(X, y, alphaValue, binarize, fitPrior, classPrior)

            print "classifier Bernoulli", classifierBernoulli

#             print "X_Testing", X_Testing
#             print "y_Testing", y_Testing
#             print classifierBernoulli.predict(X_Testing)
    #         print classifierBernoulli.score(X_Testing, y_Testing)

            accuracyUsingBernoulli =  np.mean(cross_validation.cross_val_score(classifierBernoulli, X, y, cv = cross_validate, scoring=typeOfCompare))
            print "Accuracy Using Bernoulli ", accuracyUsingBernoulli

            if accuracyUsingBernoulli == bestAccuracy:
#                 bestAccuracy = accuracyUsingBernoulli 
                bestModel = bestModel + ", Bernoulli Naive Bayes With Alpha " + str(alphaValue)
            if accuracyUsingBernoulli > bestAccuracy:
                bestAccuracy = accuracyUsingBernoulli 
                bestModel = "Bernoulli Naive Bayes With Alpha " + str(alphaValue)
            
        bernoulliExecutionTime = time.time() - bernoulliStartTime
        
        print "Bernoulli Execution Time", bernoulliExecutionTime 
        print "-- -- -- --"

        print "-- -- -- --"
        quadraticStartTime = time.time()
        print "****Quadratic Discriminant Analysis****"
        regParameterRange = [0, 0.0001, 0.001]
        for r in regParameterRange:
            print "Reg Parameter ", r
            cross_validate=cross_validation.StratifiedKFold(y, 5, shuffle=True);

            regParameter = r
            
            classifierQDA = quadraticDiscriminantAnalysis(X, y, float(regParameter))

            print "classifier QDA", classifierQDA

#             print "X_Testing", X_Testing
#             print "y_Testing", y_Testing
#             print classifierBernoulli.predict(X_Testing)
    #         print classifierBernoulli.score(X_Testing, y_Testing)

            accuracyUsingQDA =  np.mean(cross_validation.cross_val_score(classifierQDA, X, y, cv = cross_validate, scoring=typeOfCompare))
            print "Accuracy Using QDA ", accuracyUsingQDA

            if accuracyUsingQDA == bestAccuracy:
#                 bestAccuracy = accuracyUsingQDA 
                bestModel = bestModel + ", QDA With Reg_Parameter " + str(regParameter)
            if accuracyUsingQDA > bestAccuracy:
                bestAccuracy = accuracyUsingQDA 
                bestModel = "QDA With Reg_Parameter " + str(regParameter)
           
        quadraticExecutionTime = time.time() - quadraticStartTime
        
        print "Quadratic Execution Time", str(quadraticExecutionTime)
 
        print "-- -- -- --"

#         print "-- -- -- --"
        lieanerDAStartTime = time.time()
        print "****Linear Discriminant Analysis****"
        solverRange = ["svd","lsqr", "eigen"]
#         shrinkageRange = ['None', 'auto']
        
        for solve in solverRange:
            shrinkageRange = "auto"
            print "Solver", solve
            
            if solve == "svd":
                shrinkageRange = None
            
            print "Shrink", shrinkageRange
            cross_validate=cross_validation.StratifiedKFold(y, 5, shuffle=True);



            classifierLDA = linearDiscriminantAnalysis(X, y, solve, shrinkageRange)

            print "classifier LDA", classifierLDA

#             print "X_Testing", X_Testing
#             print "y_Testing", y_Testing
#             print classifierBernoulli.predict(X_Testing)
    #         print classifierBernoulli.score(X_Testing, y_Testing)

            accuracyUsingLDA =  np.mean(cross_validation.cross_val_score(classifierLDA, X, y, cv = cross_validate, scoring=typeOfCompare))
            print "Accuracy Using LDA ", accuracyUsingLDA

            
            if accuracyUsingLDA == bestAccuracy:
#                 bestAccuracy = accuracyUsingLDA 
                bestModel = bestModel + ", LDA With Solver " + str(solve) + " Shrink " + str(shrinkageRange)
            if accuracyUsingLDA >= bestAccuracy:
                bestAccuracy = accuracyUsingLDA 
                bestModel = "LDA With Solver " + str(solve) + " Shrink " + str(shrinkageRange)
            
        
        lieanerDAExecutionTime = time.time() - lieanerDAStartTime
        
        print "Linear DA Execution Time", str(lieanerDAExecutionTime)
        print "-- -- -- --"

        nearestStartTime = time.time()
        print "****Nearest Neighbor****"
        
        possibleN1 = int(len(y) * 0.01)
        if possibleN1 > 100:
            possibleN1 = 100
        if possibleN1 == 0:
            possibleN1 = 1
        neighbors = [possibleN1]
#         shrinkageRange = ['None', 'auto']
        
        for n in neighbors:
            
            print "Number Of Neighbors", n
            
            
            cross_validate=cross_validation.StratifiedKFold(y, 5, shuffle=True);

            classifierKNN = nearestNeighbor(X, y, n, "distance")

            print "classifier KNN", classifierKNN

#             print "X_Testing", X_Testing
#             print "y_Testing", y_Testing
#             print classifierBernoulli.predict(X_Testing)
    #         print classifierBernoulli.score(X_Testing, y_Testing)

            accuracyUsingKNN =  np.mean(cross_validation.cross_val_score(classifierKNN, X, y, cv = cross_validate, scoring=typeOfCompare))
        
#         cross_validation.zip
            print "Accuracy Using KNN ", accuracyUsingKNN

            if accuracyUsingKNN == bestAccuracy:
#                 bestAccuracy = accuracyUsingKNN 
                bestModel = bestModel + ", KNN With Neighbor " + str(n) + " Weight " + str("Distance")
            if accuracyUsingKNN > bestAccuracy:
                bestAccuracy = accuracyUsingKNN 
                bestModel = "KNN With Neighbor " + str(n) + " Weight " + str("Distance")
                
            
        nearestExecutionTime = time.time() - nearestStartTime
        print "Nearest Neighbor Execution Time", str(nearestExecutionTime)

        print "-- -- -- --"

        SVM_GaussianStartTime = time.time()
        print "****SVM Gaussian****"

        Cs = [0.01, 0.1, 1]
        gammas = [0.01, 0.1]
        
        for eachC in Cs:
            for eachGamma in gammas:
                cross_validate=cross_validation.StratifiedKFold(y, 5, shuffle=True);

                classifierSVM_Gaussian = SVM_Gaussian(X, y, eachC, eachGamma)
                accuracyUsingSVM_Gaussian =  np.mean(cross_validation.cross_val_score(classifierSVM_Gaussian, X, y, cv = cross_validate, scoring=typeOfCompare))


                print "Accuracy Using SVM_Gaussian "+ str(eachC) + " Gamma"  + str(eachGamma) + " Accuracy"+str(accuracyUsingSVM_Gaussian)

                if accuracyUsingSVM_Gaussian == bestAccuracy:
#                     bestAccuracy = accuracyUsingSVM_Gaussian 
                    bestModel = bestModel + ", SVM With Gaussian C " + str(eachC) + " Gamma"  + str(eachGamma)
                
                if accuracyUsingSVM_Gaussian >= bestAccuracy:
                    bestAccuracy = accuracyUsingSVM_Gaussian 
                    bestModel = "SVM With Gaussian C " + str(eachC) + " Gamma"  + str(eachGamma)
                    
                
        SVM_GaussianExecutionTime = time.time() - SVM_GaussianStartTime
        print "SVM Gaussian Execution Time", str(SVM_GaussianExecutionTime)
    print "Best Accuracy ", bestAccuracy
    print "Best Model ", bestModel

if __name__=='__main__' :
    
    randomDataset = False
    letterRecognitionDataset = True
    internetAdsDataset = True
    harsDataset = True 
    coverTypeDataset = True
    farmAdsDataset = True
    adultDataset = True
    dbWorldDataset = False #Not
    amazonDataset = False #Not
    if randomDataset:
        print "Working On RandomDataset"
        X, y = generate_data("Random")

        print "X Shape", X.shape
        print "y Shape", y.shape

        compareAccuracy(X, y, 1, True)
    if letterRecognitionDataset:
        print "Working On letterRecognitionDataset"

        X, y = generate_data("Letter Recognition")

        print "X Shape", X.shape
        print "y Shape", y.shape
        
        compareAccuracy(X, y, 1, True)
    if internetAdsDataset:
        print "Working On InternetAdsDataset"
        X, y = generate_data("InternetAdsDataset")

        print "X Shape", X.shape
        print "y Shape", y.shape
        
        compareAccuracy(X, y, 1, True)
#       
    if harsDataset:
        print "Working On UCI_HARS"
        X, y = generate_data("UCI_HARS")
    
        print "X Shape", X.shape
        print "y Shape", y.shape
        compareAccuracy(X, y, 1, False)
    if coverTypeDataset:
#         print 
        print "Working On Cover-Type"
        X, y = generate_data("cover-type")
    
        print "X Shape", X.shape
        print "y Shape", y.shape
        compareAccuracy(X, y, 1, False)
        
    if farmAdsDataset:
#         print 
        print "Working On Farm-Ads"
        X, y = generate_data("farm-ads")
    
        print "X Shape", X.shape
        print "y Shape", y.shape
        
#         print y
#         print X
        compareAccuracy(X, y, 1, False)

    if adultDataset:
        print "Working On AdultDataset"
        X, y = generate_data("adult")

        print "X Shape", X.shape
        print "y Shape", y.shape

        compareAccuracy(X, y, 1, True)
    if dbWorldDataset:
        print "Working On DBWorld"
        X, y = generate_data("DBWorld")

        print "X Shape", X.shape
        print "y Shape", y.shape
        compareAccuracy(X, y, 1, True)
        
    if amazonDataset:
        print "Working On AmazonDataset"
        X, y = generate_data("amazon")

        print "X Shape", X.shape
        print "y Shape", y.shape
        
        compareAccuracy(X, y, 1, True)
        
        
    
    doPrediction = True
    if doPrediction:
        print "Now.. For Prediction Phase:"

        randomDataset = True
        letterRecognitionDataset = True
        internetAdsDataset = True
        harsDataset = True
        coverTypeDataset = True
        farmAdsDataset = True
        adultDataset = True
        dbWorldDataset = False #Not
        amazonDataset = False #Not
        if randomDataset:
            print "Working On RandomDataset"
            X, y = generate_data("Random")

            print "X Shape", X.shape
            print "y Shape", y.shape

            X_Sub, y_Sub=randomPartition(X,y)
            print "X Sub hape", X_Sub.shape
            print "y Sub hape", y_Sub.shape
        
            compareAccuracy(X_Sub, y_Sub, 1, True)
#             compareAccuracy(X, y, 1, True)
        if letterRecognitionDataset:
            print "Working On letterRecognitionDataset"

            X, y = generate_data("Letter Recognition")

            print "X Shape", X.shape
            print "y Shape", y.shape

            X_Sub, y_Sub=randomPartition(X,y)
            print "X Sub hape", X_Sub.shape
            print "y Sub hape", y_Sub.shape
        
            compareAccuracy(X_Sub, y_Sub, 1, True)        
        if internetAdsDataset:
            print "Working On InternetAdsDataset"
            X, y = generate_data("InternetAdsDataset")

            print "X Shape", X.shape
            print "y Shape", y.shape

            X_Sub, y_Sub=randomPartition(X,y)
            print "X Sub hape", X_Sub.shape
            print "y Sub hape", y_Sub.shape
        
            compareAccuracy(X_Sub, y_Sub, 1, True)    #       
        if harsDataset:
            print "Working On UCI_HARS"
            X, y = generate_data("UCI_HARS")

            print "X Shape", X.shape
            print "y Shape", y.shape
            X_Sub, y_Sub=randomPartition(X,y)
            print "X Sub hape", X_Sub.shape
            print "y Sub hape", y_Sub.shape
        
            compareAccuracy(X_Sub, y_Sub, 1, False)
        if coverTypeDataset:
    #         print 
            print "Working On Cover-Type"
            X, y = generate_data("cover-type")

            print "X Shape", X.shape
            print "y Shape", y.shape
            X_Sub, y_Sub=randomPartition(X,y)
            print "X Sub hape", X_Sub.shape
            print "y Sub hape", y_Sub.shape
        
            compareAccuracy(X_Sub, y_Sub, 1, False)

        if farmAdsDataset:
    #         print 
            print "Working On Farm-Ads"
            X, y = generate_data("farm-ads")

            print "X Shape", X.shape
            print "y Shape", y.shape

    #         print y
    #         print X
            X_Sub, y_Sub=randomPartition(X,y)
            print "X Sub hape", X_Sub.shape
            print "y Sub hape", y_Sub.shape
        
            compareAccuracy(X_Sub, y_Sub, 1, False)

        if adultDataset:
            print "Working On AdultDataset"
            X, y = generate_data("adult")

            print "X Shape", X.shape
            print "y Shape", y.shape

            X_Sub, y_Sub=randomPartition(X,y)
            print "X Sub hape", X_Sub.shape
            print "y Sub hape", y_Sub.shape
        
            compareAccuracy(X_Sub, y_Sub, 1, True)
        if dbWorldDataset:
            print "Working On DBWorld"
            X, y = generate_data("DBWorld")

            print "X Shape", X.shape
            print "y Shape", y.shape
            compareAccuracy(X, y, 1, True)

        if amazonDataset:
            print "Working On AmazonDataset"
            X, y = generate_data("amazon")

            print "X Shape", X.shape
            print "y Shape", y.shape

            compareAccuracy(X, y, 1, True)
        
        
        
        

        
        
        
#         compareAccuracy(X, y, 1, True)


# In[ ]:



