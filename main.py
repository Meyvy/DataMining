import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

#column lables for our data set
col = [ 'buying','maint','doors','persons','lug_boot','safety','class'] 

#reading our data
df = pd.read_csv('car.data',names=col,sep=',') 

#changing categorical attributes to ordinal
enc = OrdinalEncoder()
df[['buying','maint','doors','persons','lug_boot','safety']] = enc.fit_transform(df[['buying','maint','doors','persons','lug_boot','safety']])

#choosing our predicting attribute and seperating the rest
x=df.drop('class',axis=1)
y=df[['class']]

#splitting our data into test and train partitions with the ratio of 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#creating our decision tree with no maximum or minimum restrictions(there will be no limitation for the tree and it will grow to  be a complete tree) and using gini criterion
clf_model = DecisionTreeClassifier(criterion='entropy')

#fitting the data into our model
clf_model.fit(x_train,y_train)

#calculating our tree's depth
tree_max_depth=clf_model.tree_.max_depth

#predicting the train sample 
y_predict_train=clf_model.predict(x_train)

#predicting the test sample
y_predict_test=clf_model.predict(x_test)

#printing the result
print('predicting with no maximum depth:')
print('our tree depth is',tree_max_depth,'.')
print('our model predicted the train sample with',accuracy_score(y_train,y_predict_train),'accuracy.')
print('our model predicted the test sample with',accuracy_score(y_test,y_predict_test),'accuracy.')

#using k_fold validation to determine the optimal tree depth using all the possible depths and k value 5
depth_scores=[]
for i in range(1,tree_max_depth+1):
    clf_model=DecisionTreeClassifier(criterion='entropy',max_depth=i)
    scores = cross_val_score(estimator=clf_model, X=x, y=y, cv=5, n_jobs=4) # using 4 threads
    depth_scores.append((i,scores.mean())) #calculating the avg mean

#calculating the optimal depth
depth_scores=sorted(depth_scores,key=lambda x:x[1])
optimal_depth=depth_scores[len(depth_scores)-1][0]

#printing all the depth levels and their scores
for i in range(len(depth_scores)):
    print('depth level',depth_scores[i][0],'has accuracy',depth_scores[i][1],'.')

#doing the prediction again this time with optimal depth
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
clf_model = DecisionTreeClassifier(criterion='entropy',max_depth=optimal_depth) 

#fitting the data into our model
clf_model.fit(x_train,y_train)

#predicting the train sample 
y_predict_train=clf_model.predict(x_train)

#predicting the test sample
y_predict_test=clf_model.predict(x_test)

#printing the result
print('predicting with optimal depth:')
print('our tree depth is',optimal_depth,'.')
print('our model predicted the train sample with',accuracy_score(y_train,y_predict_train),'accuracy.')
print('our model predicted the test sample with',accuracy_score(y_test,y_predict_test),'accuracy.')


