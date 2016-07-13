import csv as csv
import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import sklearn
from sklearn.metrics import roc_curve,auc,roc_auc_score

def getData(filename):
        csv_file_object = csv.reader(open(filename, 'rb'))   #Load in the csv file
        header = csv_file_object.next()                         #Skip the fist line as it is a header
        train_data=[]                                           #Create a variable called 'data'
        y = []

        for row in csv_file_object:                             #Skip through each row in the csv file
            train_data.append(row[0:-1])                          #adding each row to the data variable
            y.append(row[-1])

        train_data = np.array(train_data) #Then convert from a list to an array
        y = np.array(y)

        train_data = train_data.astype(float)
        y = y.astype(int)

        rowsright=~np.isnan(train_data).any(axis=1)
        train_data= train_data[rowsright]
        y = y[rowsright]
        return (train_data,y)

def getPrecisionRecall(y_predict,y_actual,classtag):
        y_predict =( y_predict==classtag);
        y_actual = (y_actual==classtag);
        size = len(y_predict)

        tp = 0;fp=0;fn=0;
        for i  in range(0,size):
                if(y_predict[i] == 1 and y_actual[i] == 1):
                        tp = tp+1;
                elif(y_predict[i]==1 and y_actual[i] == 0):
                        fp = fp+1;
                elif(y_predict[i] == 0 and y_actual[i] == 1):
                        fn = fn +1 ;

	#print "Values of tp,fp and fn are :",tp," : ",fp," : ",fn

        precision = 100.0*tp/(tp+fp);
        recall = 100.0*tp/(tp+fn);
        F1 = 2*precision *recall /(precision + recall) ;
	#fpr,tpr,tresholds = roc_curve(y_actual,y_predict)
	area = roc_auc_score(y_predict,y_actual)	
        return (precision,recall,F1,area)

if __name__=="__main__":
	(train_data,y)= getData("trainingData2.csv");
	(test_data,y_test)=getData("CVData2.csv");
	# Start OneVsRest Stratergy

	cOneVsRest = OneVsRestClassifier(LinearSVC()).fit(train_data,y)
	oVrPreds = cOneVsRest.predict(test_data)
	print "oVr Percentage Goods :",100.0*sum(oVrPreds == y_test)/len(y_test)
	print "Score of oVr is :",cOneVsRest.score(test_data,y_test)
	
	#scores = cross_val_score(cOneVsRest, test_data, y_test,cv=10)
	#print "Value of mean is :",scores.mean()

	# Start SVM Stratergy

	print "Started SVM Training and Classification"

	cSVM = svm.SVC().fit(train_data,y)
	SVMPreds = cSVM.predict(test_data)
	print "SVM Percentage Goods :",100.0*sum(SVMPreds == y_test) /len(y_test)

	#Start KNN Ways of prediction	

	cKNN = KNeighborsClassifier(n_neighbors=4).fit(train_data,y)
	kNNPreds = cKNN.predict(test_data)
	print "KNN Percentage Goods :",100.0*sum(kNNPreds == y_test) /len(y_test)

	#Start Decision Trees Prediction
	
	clf = tree.DecisionTreeClassifier()
	clf.fit(train_data,y)
	DTPreds = clf.predict(test_data)
	
	print "DT Percentage Goods :",100.0*sum(DTPreds == y_test)/len(y_test)

	# Started Bagging 
	clf = RandomForestClassifier(n_estimators=10)
	clf = clf.fit(train_data, y)
	RFPreds = clf.predict(test_data)

	print "RF Percentage Goods :",100.0*sum(RFPreds == y_test)/len(y_test)
	print "Version of sklearn is :",sklearn.__version__
	
	precision = np.array([0]*4)
        recall = np.array([0]*4)
        F1 = np.array([0]*4)
	area = np.array([0]*4)

	#print "SVM Preds :",SVMPreds
	#print "y_test is :",y_test
        for i,c in enumerate([1,5,6,7]):
  	      (precision[i],recall[i],F1[i],area[i]) = getPrecisionRecall(RFPreds,y_test,c)

        print "The Precision is :",precision
        print "The Recall is :",recall
        print "F1 Score is :",F1
	print "AUC is :",area

