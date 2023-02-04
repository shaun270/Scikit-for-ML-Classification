import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("diabetes.csv", sep=",")

y=df['Outcome']
x=df.drop(columns=["Outcome"])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

from xgboost import XGBClassifier

xgb_class=XGBClassifier(objective="binary:logistic", eval_metric = 'error', learning_rate = 0.1, max_depth = 4, n_estimators = 10)
xgb_class.fit(x_train,y_train)

result=xgb_class.score(x_test,y_test)
print("Accuracy : {}".format(result))

y_predict=xgb_class.predict(x_test)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,fmt='d',annot=True)
plt.show()

