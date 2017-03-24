"""
DOWNLOAD THE INCEPTION MODEL BEFORE RUNNING!
1. OPEN THE DIRECTORY WHICH CONTAINS YOUR PYTHON PACKAGAES
2. GO TO /tensorflow/models/image/imagenet
3. OPEN TERMINAL AND RUN
	 cd python classify_image.py --model_dir TUTORIAL_DIR/imagenet
4. THIS DOWNLOADS THE INCEPTION MODEL.
5. PASTE THE 'imagenet' FILE IN THE SAME FOLDER WHICH CONATAINS THIS SCRIPT.

ABOUT THE MODEL
GOOGLE INCEPTION EXTRACTS FEATURES UNTIL THE BOTTLENECT LAYER AND THEN CLASSIFIES IT ACCORDING TO THESE FEATURES.
BY REMOVING THIS LAYER, WE CAN TAKE THE FEATURE VECTOR AS OUR OUTPUT
THIS VECTOR IS THEN REDUCED FROM 2048 TO 128 USING TSNE.
"""
import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#% matplotlib inline
import pickle

model_dir = 'imagenet'
#Directory to the folder which contains image
images_dir = '/home/kushagra/Neuraltalk/'
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]

def create_graph():
	with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	labels = []
	create_graph()
	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			if (ind%100 == 0):
				print('Processing %s...' % (image))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)
			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
			print predictions.shape
			features[ind,:] = np.squeeze(predictions)
			labels.append(re.split('_\d+',image.split('/')[1])[0])
	return features, labels

features, labels = extract_features(list_images)
print features
print labels
print features.shape
# Dimensionality reduction using TSNE from 2048 to 128
model = TSNE(n_components=128, random_state=0)
np.set_printoptions(suppress=True)
print(model.fit_transform(features))
