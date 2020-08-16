import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def init():

    json_file = open('C:\\Users\\aksha\\OneDrive\\Desktop\\flood_prediction\\model\\flood_model.json','r')

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    #load woeights into new model

    loaded_model.load_weights("C:\\Users\\aksha\\OneDrive\\Desktop\\flood_prediction\\model\\flood_model.h5")

    print("Loaded Model from disk")

	#compile and evaluate loaded model

    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    #loss,accuracy = loaded_model.evaluate(x_test,y_test)
    #print('loss:', loss)

    #print('accuracy:', accuracy)
    #gets a computational graph. which is used to examine and validate the keras model this has been saved.
    graph = tf.compat.v1.get_default_graph()

    print("returning")

    return loaded_model,graph
