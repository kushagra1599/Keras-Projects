# importing the header files
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
# Importing the daaaaataset
data = pd.read_csv('heart.csv')
data.sample(frac = 1)

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
# Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Initializing the neural net
classifier = Sequential()
# Adding layers to the neural net
classifier.add(Dense(output_dim = 10,kernel_initializer= 'he_uniform', activation='relu',input_dim=13))
classifier.add(Dense(output_dim = 10,kernel_initializer= 'he_uniform', activation='relu'))
classifier.add(Dense(output_dim = 1,kernel_initializer= 'he_uniform', activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Fitting the classifier to the training set
classifier.fit(X_train,y_train,batch_size=4,epochs=100)
# Making predictions
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
