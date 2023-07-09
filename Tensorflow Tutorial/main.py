import pandas as pd
import tensorflow as tf
from tensorflow import keras
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


train = train.drop(["Precip Type","Formatted Date","Summary","Apparent Temperature (C)","Wind Bearing (degrees)","Loud Cover","Daily Summary"],axis = 1)
test = test.drop(["Precip Type","Formatted Date","Summary","Apparent Temperature (C)","Wind Bearing (degrees)","Loud Cover","Daily Summary"],axis = 1)

train_y= train.drop(["Humidity", "Wind Speed (km/h)" , "Visibility (km)"  ,"Pressure (millibars)"],axis = 1)
test_y  = test.drop(["Humidity", "Wind Speed (km/h)" , "Visibility (km)"  ,"Pressure (millibars)"],axis = 1)

train_x = train.drop("Temperature (C)",axis = 1)
test_x = test.drop("Temperature (C)",axis = 1)

print(train_x)
print(test_x)

print(train_y)
print(test_y)



model = keras.Sequential()
model.add(keras.layers.Dense(4,activation=tf.nn.relu))
model.add(keras.layers.Dense(2,activation=tf.nn.relu))
model.add(keras.layers.Dense(1))

optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss="mse",optimizer=optimizer,metrics=["mae","mse"])
history = model.fit(x=train_x,y=train_y,batch_size=100, epochs=10)
import matplotlib.pyplot as plt

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.plot(history.epoch,history.history['loss'])
plt.show()

loss, mae,mse = model.evaluate(test_x,test_y,verbose=0)

plt.figure()
plt.bar(["loss","mae","mse"],[loss,mae,mse],width = 0.4,color='red')
plt.show()


