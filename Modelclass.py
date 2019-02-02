import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import random
from sklearn.metrics import accuracy_score

class Modelcnn():

    def __init__(self,x_train,y_train,x_test,y_test,neuron1,neuron2,neuron3,neuron4,act1,act2,act3,act4,opt,bs,ep):
        self.neuron1=neuron1
        self.neuron2=neuron2
        self.neuron3=neuron3
        self.neuron4=neuron4
        self.act1=act1
        self.act2=act2
        self.act3=act3
        self.act4=act4
        self.opt=opt
        self.bs=bs
        self.ep=ep
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.model=None
        self.preds=None
        self.predround=None

    def createModel(self,):
        self.model = Sequential()
        self.model.add(Conv2D(self.neuron1, kernel_size=(3,3), activation = self.act1, input_shape=(28, 28 ,1) ))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))

        self.model.add(Conv2D(self.neuron2, kernel_size = (3, 3), activation = self.act2))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))

        self.model.add(Conv2D(self.neuron3, kernel_size = (3, 3), activation = self.act3))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.neuron4, activation = self.act4))
        self.model.add(Dropout(0.20))
        self.model.add(Dense(24, activation = 'softmax'))

        self.model.compile(loss = keras.losses.categorical_crossentropy, optimizer=self.opt,metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs=self.ep, batch_size=self.bs)

    def predict(self,test_images,test_labels):
        y_pred = self.model.predict(test_images)
        self.preds = y_pred
        self.predround = y_pred.round()
        return(accuracy_score(test_labels, self.predround))
