import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
import os

import timeit

n_components=3
t_size=0.2


df = pd.read_csv('phising data with description.txt')
#print("Full data: ")
#print(df)

#print("df size before drop: ")
#print(df.shape)
#print("Column 1: ")
#print(df['c1'])

#print("Result data: ")
#print(df['result'])



#print("y: ")
#print(y)
#print(y.shape)
#print(X.shape)


#print("df size after drop: ")
#print(df.shape)

f = open('pca components.txt','w')

true=1
while true:
    X=np.array(df.drop(['result'],1))
    #print("X: ")
    #print(X)
    y=np.array(df['result'])
    
    X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=t_size)

    # Start timer
    


    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    X_test_alone=[[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]]
    start = timeit.default_timer()
    y_predict_alone=clf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction original KNN: ",(stop - start)*1000 ) 

    
    print("Alone prediction is: ",y_predict_alone)
    
    #find the accuracy for Original phishing
    accuracy_original = clf.score(X_test,y_test)
    print("Accuracy for Original Phishing KNN: ")
    print(accuracy_original)


    #find null accuracy for original dataset

    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)
    
    y_predict=clf.predict(X_test)
    
    
    #print(tprk)
    #print(fprk)
    #print(tprk.shape)

    #for i in range(0,fprk.shape[0]):
        #print(fprk[i]," ",tprk[i])
    
    #plt.show()
   

    

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for original dataset: ",count.head(1)[1]/2211)
    print("Accuracy for original by confusion matrix: ",acc)
    print("Error rate for original KNN: ",err)
    print("Sensitivity  for original KNN: ",sen)
    print("Specificity  for original KNN: ",spe)
    print("Precision  for original KNN: ",pre)
    print("Recall  for original KNN: ",rec)
    print("F-Score  for original KNN: ",fsc)
    print("TPR  for original KNN: ",tpr)
    print("FPR  for original KNN: ",fpr)
    print()

    
    
    
#original SVM
    
    clf_svm=svm.SVC()
    clf_svm.fit(X_train,y_train)

    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf_svm.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Time for modeling and prediction original SVM: ",(stop - start)*1000 )
    
    #find the accuracy for SVM
    accuracy_pca_svm = clf_svm.score(X_test,y_test)
    print("Accuracy for Original svm: ")
    print(accuracy_pca_svm)


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_svm.predict(X_test)



    
    #print(tprk)
    #print(fprk)
    #print(tprk.shape)

    #for i in range(0,fprk.shape[0]):
        #print(fprk[i]," ",tprk[i])
    
    #plt.show()

    

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for original dataset : ",count.head(1)[1]/2211)
    print("Accuracy for original SVM: ",acc)
    print("Error rate for original SVM: ",err)
    print("Sensitivity  for original SVM: ",sen)
    print("Specificity  for original SVM: ",spe)
    print("Precision  for original SVM: ",pre)
    print("Recall  for original SVM: ",rec)
    print("F-Score  for original SVM: ",fsc)
    print("TPR  for original SVM: ",tpr)
    print("FPR  for original SVM: ",fpr)
    print()


    

    #original + Random Forest
    from sklearn.ensemble import RandomForestClassifier
    
    
    
    clf_rf=RandomForestClassifier(n_estimators=1) 
    clf_rf.fit(X_train,y_train)
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf_rf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction original RF: ",(stop - start)*1000 )

    #find the accuracy for SVM
    accuracy_pca_rf = clf_rf.score(X_test,y_test)
    print("Accuracy for Originallll & RF: ")
    print(accuracy_pca_rf)


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_rf.predict(X_test)

    
    #print(tprk)
    #print(fprk)
    #print(tprk.shape)

    #for i in range(0,fprk.shape[0]):
        #print(fprk[i]," ",tprk[i])
    
    #plt.show()

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for pca dataset : ",count.head(1)[1]/2211)
    print("Accuracy for original RF: ",acc)
    print("Error rate for original RF: ",err)
    print("Sensitivity  for original RF: ",sen)
    print("Specificity  for original RF: ",spe)
    print("Precision  for original RF: ",pre)
    print("Recall  for original RF: ",rec)
    print("F-Score  for original RF: ",fsc)
    print("TPR  for original RF: ",tpr)
    print("FPR  for original RF: ",fpr)
    print()

    #t_size = t_size + 0.05

    #if t_size > 0.2:
        #break
    #Original Naive Bayes

    
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    
    accuracy_nb = clf.score(X_test,y_test)
    print("Accuracy for Original & Naive Bayes: ",accuracy_nb)
    #print(accuracy_nb)
    




    #PCA
    from sklearn.decomposition import PCA
    
    start = timeit.default_timer()
    
    scaler = StandardScaler()
    scaler.fit(df.drop(['result'],1))
    scaled_data = scaler.transform(df.drop(['result'],1))
    #print("Scaled data: ",scaled_data)


    
    pca = PCA(n_components)

    # fragmenting in components, final result is x_pca
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    #print("Size of data for original and after pca")
    #print(scaled_data.shape)
    print(x_pca.shape)

    #print("data after PCA: ")
    #print(x_pca)

    #plot the orthoganal components in X Y axis

    #plt.figure(figsize=(10,10))
    #plt.scatter(x_pca[:,0],x_pca[:,1],c=y,cmap='plasma')
    #plt.xlabel('First Component')
    #plt.ylabel('Second Component')
    #plt.zlabel('Third Component')

    
    figa=plt.figure()
    ax=figa.add_subplot(111,projection='3d')

    X=x_pca[:,0]
    Y=x_pca[:,1]
    Z=x_pca[:,2]

    ax.scatter(X,Y,Z,c='r',marker='o')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    plt.show()
    
    print("PCA component 1:")
    for z in range(0,11055):
        #print(x_pca[:,0][z])
        line=str(x_pca[:,0][z])+"\t"+str(x_pca[:,1][z])+"\t"+str(x_pca[:,2][z])+"\n"
        f.write(line)
    print(x_pca[:,0])
    #f.write(x_pca[:,0])
    f.close()

    #plt.show()

    #fit to KNN
    X=x_pca
    #print("X for knn: ")
    #print(X)


    y=np.array(y)
    stop = timeit.default_timer()
    pre_time=stop-start
    #print ("y for knn: ")
    #print(y)
    #print("changed data:")
    #print(type(x_pca))
    #print(type(y))



    # import necessary things for KNN and accuracy
    

    #print("Test")
    startp = timeit.default_timer()
    X_test_alon=df.drop(['result'],1).head(n_components)
    #Scaler = StandardScaler()
    scaler.fit(X_test_alon)
    scaled_data = scaler.transform(X_test_alon)
    #print("Our test alone scaled data: ",scaled_data)
    pca.fit(scaled_data)
    X_test_alone = pca.transform(scaled_data)
    stopp = timeit.default_timer()
    pcaconersion=stopp-startp
    print("PCA conversion time: ",stopp-startp)

    X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=t_size)

    
    clf=neighbors.KNeighborsClassifier()

    
    
    clf.fit(X_train,y_train)
    
    #X_test_alon=[[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]]
    
    #print("X_test_alone: \n",X_test_alone)
    start = timeit.default_timer()
    y_predict_alone=clf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction PCA KNN: ",((stop - start)/n_components)*1000)

    #find the accuracy for KNN
    accuracy_pca = clf.score(X_test,y_test)
    print("Accuracy for PCA KNN: ")
    print(accuracy_pca)

    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fprk, tprk, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fprk, tprk)

    print("AUC PCA KNN : ",roc_auc)

    # method I: plt
    
    plt.title('Receiver Operating Characteristic - PCA')
    plt.plot(fprk, tprk, 'r', label = 'AUC- PCA+KNN = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print("PCA knn")
    #print(tprk)
    #print(fprk)
    #print(tprk.shape)

    #for i in range(0,fprk.shape[0]):
        #print(fprk[i]," ",tprk[i])
    
    #plt.show()
    


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for pca dataset: ",count.head(1)[1]/2211)
    print("Accuracy for pca by confusion matrix: ",acc)
    print("Error rate for pca KNN: ",err)
    print("Sensitivity  for pca KNN: ",sen)
    print("Specificity  for pca KNN: ",spe)
    print("Precision  for pca KNN: ",pre)
    print("Recall  for pca KNN: ",rec)
    print("F-Score  for pca KNN: ",fsc)
    print("TPR  for pca KNN: ",tpr)
    print("FPR  for pca KNN: ",fpr)
    print()



    #PCA + SVM
    
    clf_svm=svm.SVC(probability=True)
    clf_svm.fit(X_train,y_train)
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf_svm.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction PCA SVM: ",((stop - start)/n_components)*1000)

    
    #find the accuracy for SVM
    accuracy_pca_svm = clf_svm.score(X_test,y_test)
    print("Accuracy for PCA svm: ")
    print(accuracy_pca_svm)

    probs = clf_svm.predict_proba(X_test)
    preds = probs[:,1]
    fprk, tprk, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fprk, tprk)

    print("AUC PCA SVM: ",roc_auc)

    # method I: plt
    
    plt.title('Receiver Operating Characteristic - PCA')
    plt.plot(fprk, tprk, 'g', label = 'AUC- PCA+SVM = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print("PCA SVM")
    


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_svm.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for pca dataset : ",count.head(1)[1]/2211)
    print("Accuracy for pca SVM: ",acc)
    print("Error rate for pca SVM: ",err)
    print("Sensitivity  for pca SVM: ",sen)
    print("Specificity  for pca SVM: ",spe)
    print("Precision  for pca SVM: ",pre)
    print("Recall  for pca SVM: ",rec)
    print("F-Score  for pca SVM: ",fsc)
    print("TPR  for pca SVM: ",tpr)
    print("FPR  for pca SVM: ",fpr)
    print()
    


    #PCA + Random Forest
    
    from sklearn.ensemble import RandomForestClassifier
    
    
    clf_rf=RandomForestClassifier(n_estimators=100) 
    clf_rf.fit(X_train,y_train)
    start = timeit.default_timer()

    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    y_predict_alone=clf_rf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction PCA RF: ",((stop - start)/n_components)*1000)
    
    #find the accuracy for SVM
    accuracy_pca_rf = clf_rf.score(X_test,y_test)
    print("Accuracy for PCA & RF: ")
    print(accuracy_pca_rf)

    probs = clf_rf.predict_proba(X_test)
    preds = probs[:,1]
    fprk, tprk, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fprk, tprk)

    print("AUC PCA RF: ",roc_auc)

    # method I: plt
    
    plt.title('Receiver Operating Characteristic - PCA')
    plt.plot(fprk, tprk, 'b', label = 'AUC- PCA+RF = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print("PCA RF")


    plt.show()

    n_components=n_components+1
    #if n_components<10:
        #continue
    if n_components < n_components+1:
        break
    
    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_rf.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for pca dataset : ",count.head(1)[1]/2211)
    print("Accuracy for pca RF: ",acc)
    print("Error rate for pca RF: ",err)
    print("Sensitivity  for pca RF: ",sen)
    print("Specificity  for pca RF: ",spe)
    print("Precision  for pca RF: ",pre)
    print("Recall  for pca RF: ",rec)
    print("F-Score  for pca RF: ",fsc)
    print("TPR  for pca RF: ",tpr)
    print("FPR  for pca RF: ",fpr)
    print()


    #PCA + Linear regression
    from sklearn import linear_model
    print("n_components: ",n_components)
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X_train,y_train)
    #find the accuracy for linear regression
    accuracy_pca_lr = linear_regression.score(X_test,y_test)
    print("Accuracy for PCA & LR: ")
    print(accuracy_pca_lr)







    #PCA + Logistic regression
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2',C=1)
    model.fit(X_train,y_train)

    #find the accuracy for Logistic regression
    accuracy_pca_logr = model.score(X_test,y_test)
    print("Accuracy for PCA & Logistic R: ")
    print(accuracy_pca_logr)


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=model.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for pca dataset : ",count.head(1)[1]/2211)
    print("Accuracy for pca LogR: ",acc)
    print("Error rate for pca LogR: ",err)
    print("Sensitivity  for pca LogR: ",sen)
    print("Specificity  for pca LogR: ",spe)
    print("Precision  for pca LogR: ",pre)
    print("Recall  for pca LogR: ",rec)
    print("F-Score  for pca LogR: ",fsc)
    print("TPR  for pca LogR: ",tpr)
    print("FPR  for pca LogR: ",fpr)
    print()


    #PCA Naive Bayes
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    
    accuracy_nb = clf.score(X_test,y_test)
    print("Accuracy for PCA & Ber. Naive Bayes: ",accuracy_nb)

    #Random Gaussian Projection


    from sklearn import random_projection
    
    
    
    transformer = random_projection.GaussianRandomProjection(n_components)
    #print("Transformer matrix: ")
    #print(transformer)
    X = transformer.fit_transform(df.drop(['result'],1))

    startg = timeit.default_timer()
    transformer1 = random_projection.GaussianRandomProjection(n_components)
    X_test_alone=[[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]]
    X_test_alone=transformer1.fit_transform(X_test_alone)
    stopg = timeit.default_timer()
    convtime=stopg-startg

    
    y=np.array(df['result'])
    
    #stop = timeit.default_timer()
    

    #find accuracy for random gaussian Projection
    X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=t_size)

    
    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction Gaussian KNN: ",(stop - start)*1000)
    #find the accuracy for RGP
    accuracy_gaussian = clf.score(X_test,y_test)
    print("Accuracy for gaussian KNN: ")
    print(accuracy_gaussian)


    #Gaussian KNN
    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for original dataset: ",count.head(1)[1]/2211)
    print("Accuracy for Gaussian by confusion matrix: ",acc)
    print("Error rate for Gaussian KNN: ",err)
    print("Sensitivity  for Gaussian KNN: ",sen)
    print("Specificity  for Gaussian KNN: ",spe)
    print("Precision  for Gaussian KNN: ",pre)
    print("Recall  for Gaussian KNN: ",rec)
    print("F-Score  for Gaussian KNN: ",fsc)
    print("TPR  for Gaussian KNN: ",tpr)
    print("FPR  for Gaussian KNN: ",fpr)
    print()




    #Gaussian SVM

    
    clf_svm=svm.SVC()
    clf_svm.fit(X_train,y_train)
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf_svm.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction Gaussian SVM: ",(stop - start)*1000)
    #find the accuracy for SVM
    accuracy_gau_svm = clf_svm.score(X_test,y_test)
    print("Accuracy for Gaussian svm: ")
    print(accuracy_gau_svm)


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_svm.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for Gaussian dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for pca dataset : ",count.head(1)[1]/2211)
    print("Accuracy for Gaussian SVM: ",acc)
    print("Error rate for Gaussian SVM: ",err)
    print("Sensitivity  for Gaussian SVM: ",sen)
    print("Specificity  for Gaussian SVM: ",spe)
    print("Precision  for Gaussian SVM: ",pre)
    print("Recall  for Gaussian SVM: ",rec)
    print("F-Score  for Gaussian SVM: ",fsc)
    print("TPR  for Gaussian SVM: ",tpr)
    print("FPR  for Gaussian SVM: ",fpr)
    print()


    #Gausian + RF

    
    
    clf_rf=RandomForestClassifier(n_estimators=100) 
    clf_rf.fit(X_train,y_train)
    start = timeit.default_timer()
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    y_predict_alone=clf_rf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction Gaussian RF: ",(stop - start)*1000)
    #find the accuracy for Gaussian RF
    accuracy_Gau_rf = clf_rf.score(X_test,y_test)
    print("Accuracy for Gausian & RF: ")
    print(accuracy_Gau_rf)


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_rf.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size) 
    print("null accuracy for Original dataset : ",count.head(1)[1]/2211)
    print("Accuracy for Gausian RF: ",acc)
    print("Error rate for Gausian RF: ",err)
    print("Sensitivity  for Gausian RF: ",sen)
    print("Specificity  for Gausian RF: ",spe)
    print("Precision  for Gausian RF: ",pre)
    print("Recall  for Gausian RF: ",rec)
    print("F-Score  for Gausian RF: ",fsc)
    print("TPR  for pca Gausian: ",tpr)
    print("FPR  for pca Gausian: ",fpr)
    print()








    #Random Sparse Projection
    
    start = timeit.default_timer()
    
    transformer = random_projection.SparseRandomProjection(n_components)
    X = transformer.fit_transform(df.drop(['result'],1))
    #X_test_alone=transformer.fit_transform(X_test_alone)
    y=df['result']
    
    stop = timeit.default_timer()
    pre_time=stop-start

    starts = timeit.default_timer()
    transformer1 = random_projection.SparseRandomProjection(n_components)
    X_test_alone=[[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]]
    X_test_alone=transformer1.fit_transform(X_test_alone)
    stops = timeit.default_timer()
    convtime=stops-starts
    
    #find accuracy for random Sparse Projection
    X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=t_size)

    
    
    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction Sparse KNN: ",(stop - start)*1000)


    #find the accuracy for RSP and KNN
    accuracy_sparse = clf.score(X_test,y_test)
    print("Accuracy for Sparse Projection KNN: ")
    print(accuracy_sparse)

    #Sparse KNN
    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for original dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for original dataset: ",count.head(1)[1]/2211)
    print("Accuracy for Sparse by confusion matrix: ",acc)
    print("Error rate for Sparse KNN: ",err)
    print("Sensitivity  for Sparse KNN: ",sen)
    print("Specificity  for Sparse KNN: ",spe)
    print("Precision  for Sparse KNN: ",pre)
    print("Recall  for Sparse KNN: ",rec)
    print("F-Score  for Sparse KNN: ",fsc)
    print("TPR  for Sparse KNN: ",tpr)
    print("FPR  for Sparse KNN: ",fpr)
    print()




    
    clf_svm=svm.SVC()
    clf_svm.fit(X_train,y_train)
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf_svm.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction Sparse SVM: ",(stop - start)*1000)


    #find the accuracy for RSP SVM

    accuracy_spa_svm = clf_svm.score(X_test,y_test)
    print("Accuracy for Sparse svm: ")
    print(accuracy_spa_svm)


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_svm.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for Sparse dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for Original dataset : ",count.head(1)[1]/2211)
    print("Accuracy for Sparse SVM: ",acc)
    print("Error rate for Sparse SVM: ",err)
    print("Sensitivity  for Sparse SVM: ",sen)
    print("Specificity  for Sparse SVM: ",spe)
    print("Precision  for Sparse SVM: ",pre)
    print("Recall  for Sparse SVM: ",rec)
    print("F-Score  for Sparse SVM: ",fsc)
    print("TPR  for Sparse SVM: ",tpr)
    print("FPR  for Sparse SVM: ",fpr)
    print()


    #Sparse and RF
    
    
    
    clf_rf=RandomForestClassifier(n_estimators=100) 
    clf_rf.fit(X_train,y_train)
    #X_test_alone=[-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,-1]
    start = timeit.default_timer()
    y_predict_alone=clf_rf.predict(X_test_alone)

    #End Timer
    stop = timeit.default_timer()
    print("Total time taken for  modeling and prediction Sparse RF: ",(stop - start)*1000)
    #find the accuracy for Gaussian RF
    accuracy_spa_rf = clf_rf.score(X_test,y_test)
    print("Accuracy for Sparse & RF: ")
    print(accuracy_spa_rf)


    yt=pd.Series(y_test)
    #print(yt.value_counts())
    count=yt.value_counts()
    #print(max(yt.mean(),1-yt.mean()))
    #print(count.head(1)[1]/2211)

    y_predict=clf_rf.predict(X_test)

    conf=metrics.confusion_matrix(y_test,y_predict)
    print("Confusion matrix for Sparse dataset: ")
    print(conf)

    TP=conf[1,1]
    TN=conf[0,0]
    FP=conf[0,1]
    FN=conf[1,0]

    acc=(TP+TN)/(TP+TN+FP+FN)
    err=(1-acc)
    sen=TP/(TP+FN)
    spe=TN/(FP+TN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    fsc=(2*pre*rec)/(pre+rec)
    tpr=sen
    fpr=(1-spe)

    print()
    print("n_components: ",n_components)
    print("test size: ",t_size)
    print("null accuracy for Original dataset : ",count.head(1)[1]/2211)
    print("Accuracy for Sparse RF: ",acc)
    print("Error rate for Sparse RF: ",err)
    print("Sensitivity  for Sparse RF: ",sen)
    print("Specificity  for Sparse RF: ",spe)
    print("Precision  for Sparse RF: ",pre)
    print("Recall  for Sparse RF: ",rec)
    print("F-Score  for Sparse RF: ",fsc)
    print("TPR  for pca Sparse: ",tpr)
    print("FPR  for pca Sparse: ",fpr)
    print()



    #print("test size: ",t_size)
    
    
    
    

    print()
    print()
    print()

    print("feature : ",n_components)
    print("Original: ",accuracy_original)
    print("PCA     : ",accuracy_pca)
    print("Sparse  : ",accuracy_sparse)
    print("Gaussian: ",accuracy_gaussian)

    print()
    print("pca & knn: ",accuracy_pca)
    print("pca & svm: ",accuracy_pca_svm)
    print("pca & rf: ",accuracy_pca_rf)
    print("pca & lr: ",accuracy_pca_lr)
    print("pca & logR: ",accuracy_pca_logr)

    




















