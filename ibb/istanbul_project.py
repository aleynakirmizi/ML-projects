import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve,cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
dataset = pd.read_excel('ibb-dataset-proje.xlsx')
df = dataset.copy()
date_default = datetime.datetime(2015,11,4)
date_list=[]
for date in df["SUBSCRIPTION_DATE"]:
    diff = date - date_default
    diff = diff.days
    date_list.append(diff)
df["DATE_DIFF"] = pd.DataFrame(date_list)
day_list = list()
for date in df["SUBSCRIPTION_DATE"]:
    date_c = datetime.datetime.ctime(date)[0:2]
    day_list.append(date_c)
day_list_array = np.array(day_list)
day_list_df = pd.DataFrame(day_list_array)
df["DAYS"] = day_list_df
df["NEW_DAYS"] = np.where(df["DAYS"].str.contains("Su|Sa"),0,1)
month_list=[]
for date in df["SUBSCRIPTION_DATE"]:
  month = date.strftime('%B')
  month_list.append(month)
df["MONTHS"] = month_list
le = preprocessing.LabelEncoder()
df["NEW_MONTHS"]=le.fit_transform(df["MONTHS"])
unknown = df[df.SUBSCRIBER_DOMESTIC_FOREIGN=='Bilinmiyor']
unknown_array = np.array(unknown.index)
df.drop(index=unknown_array,inplace=True)
istanbul = df[df.SUBSCRIPTION_COUNTY == 'İSTANBUL']
istanbul_array=np.array(istanbul.index)
df.drop(index=istanbul_array,inplace = True)
columns = ["LONGITUDE","LATITUDE"]
df.drop(columns=columns,inplace=True)
df["NEW_SUBSCRIBER_DOMESTIC_FOREIGN"]=np.where(df["SUBSCRIBER_DOMESTIC_FOREIGN"].str.contains("Yerli"),1,0)
df = pd.get_dummies(df,columns=["SUBSCRIPTION_COUNTY"],prefix=["SUBSCRIPTION_COUNTY"])
df["NEW_DAYS"] = np.where(df["DAYS"].str.contains('Su|Sa'),0,1)
df = df.reset_index(drop=True)
columns=["_id","SUBSCRIPTION_DATE","SUBSCRIBER_DOMESTIC_FOREIGN","DAYS","MONTHS"]
df.drop(columns=columns,inplace=True)
y=df["NUMBER_OF_SUBSRIBER"]
x=df.drop(columns="NUMBER_OF_SUBSRIBER")
model = RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_split=8,min_samples_leaf=4,max_features='auto')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=80)
model.fit(x_train,y_train)
pred = model.predict(x_test)
print("r2:",r2_score(y_test,pred))
print("mse:",mean_squared_error(y_test,pred))
print("mean_absolute_error:",mean_absolute_error(y_test,pred))
plt.figure(figsize=(10,10))
xax=sns.distplot(y,hist=False,color="r",label="Actual Value")
sns.distplot(pred,hist=False,color="b",label="Fitted Values",ax=xax)
plt.title("Actual vs Fitted Values")
plt.show()
