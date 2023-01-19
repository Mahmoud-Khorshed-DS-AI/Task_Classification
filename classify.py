import argparse
import PIL
import tensorflow as tf
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow_hub as hub
import argparse
import re
import numpy as np


def classify(model_path, img_link):
    class_names = ['bus', 'crossover', 'hatchback', 'motorcycle', 'pickup-truck', 'sedan', 'truck', 'van']
    model = tf.keras.models.load_model(model_path)
    img_path = tf.keras.utils.get_file('Red_sunflower', origin=img_link)
    img = tf.keras.utils.load_img( img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[predictions[0]], 100 * np.max(score)))


# Initializing Parser
parser = argparse.ArgumentParser(description ='sort some integers.')

# Adding Argument
parser.add_argument('model_path',metavar ='N',type = str,nargs ='+',help ='model path')
parser.add_argument('img_path',metavar ='N',type = str,nargs ='+',help ='img path')

args = parser.parse_args()

classify(args.model_path[0], args.img_path[0])
