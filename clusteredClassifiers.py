from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier #,BaggingClassifier
import classifiers
import csv
import numpy as np

no_predictors = 4

def getClusterifiedTrainingData(filename,d):
	csv_file_object = csv.reader(open(filename, 'rb'))   #Load in the csv file
        header = csv_file_object.next()  
	
	train_data=[[],[],[],[],[],[],[],[],[],[]]	

	(x,y)=classifiers.getData(filename)
	for (row,val) in zip(x,y):
		train_data[int(val)].append(row)
	
	topGenresTrainData = []	

	for i in range(0,no_predictors):
		topGenresTrainData.append([])

	for (key,value) in d.iteritems():
		topGenresTrainData[value].append(train_data[key])

	return (train_data,topGenresTrainData)	

def getMainGenrePredictor(toGenresTrainingData):
        topGenresY=[]
	for i in range(0,no_predictors):
		topGenresY.append([i]*len(topGenresTrainData[i]))

        train_data1 = [item for sublist in topGenresTrainData for item in sublist]
        train_y = [item for sublist in topGenresY for item in sublist]
	#bgg = BaggingClassifier(svm.SVC(kernel="linear"), n_estimators=500)
	#bdt = AdaBoostClassifier(svm.SVC(kernel='linear'), algorithm="SAMME",n_estimators=200)
	#bgg.fit(train_data1,train_y)

        #Train our 4 Important Genres
        cSVM = svm.SVC(kernel='linear').fit(train_data1,train_y)
        #clf = RandomForestClassifier(n_estimators=10)
	#clf = clf.fit(train_data1, train_y)

        return (cSVM,train_data1,train_y)


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
	precision = 100.0*tp/(tp+fp);
	recall = 100.0*tp/(tp+fn);
	F1 = 2*precision *recall /(precision + recall) ;
	
	return (precision,recall,F1)

def getResultsTransformed(y_test,mapping):
	y_test2 =  y_test.copy()
	for i in range(0,len(y_test)):
		y_test2[i]=mapping[y_test[i]]	
	return y_test2	


if __name__=="__main__":
	init_mapping={0:1,1:2,2:1,3:0,4:3,5:2,6:0,7:0,8:3,9:0}
	#init_mapping={0:0,1:1,2:2,3:3,4:0,5:1,6:4,7:0,8:2,9:0}
	#init_mapping={0:0,1:1,2:0,3:2,4:0,5:1,6:2,7:2,8:0,9:2}
	#init_mapping={0:0,1:1,2:2,3:3,4:0,5:1,6:2,7:3,8:2,9:3}	

        (train_data,topGenresTrainData)=getClusterifiedTrainingData("trainingData.csv",init_mapping)
	
	step2y=[]
	cSVMmain = []

	for i in range(0,no_predictors):
		step2y.append([])		
		cSVMmain.append([])		

	for (key,val) in init_mapping.iteritems():
		step2y[val].append([key]*len(train_data[key]))		

	for i in range(0,no_predictors):
	        topGenresTrainData[i] = [item for sublist in topGenresTrainData[i] for item in sublist]
		step2y[i] = [item for sublist in step2y[i] for item in sublist]

	
        (MainPredictor,train_data1,train_y1) = getMainGenrePredictor(topGenresTrainData);
        (test_data,y_test) = classifiers.getData("CVData.csv")


	#cSVM0 = svm.SVC().fit(np.array(topGenresTrainData[0]),step2y[0])
	#cSVM1 = svm.SVC().fit(np.array(topGenresTrainData[1]),step2y[1])
	
	for i in range(0,no_predictors):
		#print "Value of i :",i," step2y[i]:",step2y[i]
		try:
			cSVMmain[i]=svm.SVC(kernel='linear').fit(np.array(topGenresTrainData[i]),step2y[i])
	        	#cSVMmain[i] = BaggingClassifier(svm.SVC(kernel='rbf'),n_estimators=1000)
			#cSVMmain[i] = AdaBoostClassifier(svm.SVC(kernel='linear'), algorithm="SAMME",n_estimators=200)
			#print "Done Boosting for Classifier :",i
			#cSVMmain[i] = BaggingClassifier(base_estimator=cSVMmain1[i],n_estimators=10)
			#cSVMmain[i].fit(np.array(topGenresTrainData[i]),step2y[i])
			#print "Done Boosting and Bagging for Classifier ",i
		except Exception,ex:
			print "for following i, NO TRAINING: ",i,".....",ex
			cSVMmain[i] = step2y[i][0]

	#cSVM0 =  RandomForestClassifier(n_estimators=10).fit(np.array(topGenresTrainData[1]),step2y[1])
	#cSVM1 =  RandomForestClassifier(n_estimators=10).fit(np.array(topGenresTrainData[1]),step2y[1])
	#cSVM2 =  RandomForestClassifier(n_estimators=10).fit(np.array(topGenresTrainData[1]),step2y[1])
	#cSVM3 =  RandomForestClassifier(n_estimators=10).fit(np.array(topGenresTrainData[1]),step2y[1])
	
	#(MainPredictor,train_data1,train_y1,RFClassifier) = getMainGenrePredictor(topGenresTrainData);
	#(test_data,y_test) = classifiers.getData("CVData.csv")
	#neuralnetsResults = neuralnets.neuralnetsTraining(train_data1,train_y1,test_data,y_test)
	
	#(test_data,y_test) = classifiers.getData("CVData.csv")

	MainPredictions = MainPredictor.predict(test_data)
	print "Got Level 1 Predictions"
	#MainPredictions = RFClassifier.predict(test_data)
	ylevel1 = getResultsTransformed(y_test,init_mapping)
	print "First level accuracy :",100.0*sum(ylevel1 == MainPredictions)/len(ylevel1)
	
	#for i in range(0,no_predictors):
        #	cSVMmain[i] = svm.SVC(kernel='linear').fit(np.array(topGenresTrainData[i]),np.array(step2y[i]))

			
	for (i,pred) in enumerate(MainPredictions):
		k = int(pred)
		try:
			#print "Value of Pred is :",cSVMmain[k].predict(test_data[i]),"---------\n"
			MainPredictions[i] = int(cSVMmain[k].predict(test_data[i])[0])
		except Exception,ex:
			print "The Error is :",ex, ".........",k,"......",i
			MainPredictions[i] = int(cSVMmain[k])


	#print "Main Predictions here :",MainPredictions
	
		#if(int(pred)==0):
		#	MainPredictions[i] = int(cSVM0.predict(test_data[i])[0])
		#elif(int(pred)==1):
		#	MainPredictions[i] = int(cSVM1.predict(test_data[i])[0])
		#elif(int(pred)==2):
                #        MainPredictions[i] = int(cSVM1.predict(test_data[i])[0])
		#elif(int(pred)==3):
                #        MainPredictions[i] = int(cSVM1.predict(test_data[i])[0])

	try:	
		precision = np.array([0]*10)
		recall = np.array([0]*10)
		F1 = np.array([0]*10)

		for c in range(0,10):
			(precision[c],recall[c],F1[c]) = getPrecisionRecall(MainPredictions,y_test,c)
	
		print "The Precision is :",precision
		print "The Recall is :",recall
		print "F1 Score is :",F1
	except Exception,ex:
		print "the error is :",c,".....",ex

	print MainPredictions,len(MainPredictions)
	#print y_test
	print "The Accuracy of this Process :",100.0*sum(MainPredictions==y_test)/len(y_test)
	
def getMainGenrePredictor(toGenresTrainingData,topGenresY):
	topGenresY=[]
	topGenresY.append([0]*len(topGenresTrainData[0]))	
	topGenresY.append([1]*len(topGenresTrainData[1]))
	topGenresY.append([2]*len(topGenresTrainData[2]))
	topGenresY.append([3]*len(topGenresTrainData[3]))

	train_data1 = [item for sublist in topGenresTrainData for item in sublist]	
	train_y1 = [item for sublist in topGenresY for item in sublist]
	
	#Train our 4 Important Genres
	cSVM = svm.SVC().fit(train_data1,train_y1)
	level1Predds = cSVM.predict(train_data1)
	#print 
	return (cSVM,train_data1,train_y1)	
