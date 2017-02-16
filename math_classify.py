# set up Python environment: numpy for numerical routines, and matplotlib for plotitng
from PIL import Image
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.figure as fig
#display plots in this notebook
from IPython import get_ipython
#import ipdb

#set display defaults
plt.rcParams['figure.figsize'] = (10, 10)	# large images
plt.rcParams['image.interpolation'] = 'nearest'	# don't interpolate: show square pixels
plt.rcParams['image.cmap']='gray' #use grayscale output rather than a color heatmap

# The caffe module needs to be on the Python path;
# we'll add it here explicityly

import sys
caffe_root = '../' # this file should be run from {caffe_root}/
sys.path.insert(0, caffe_root + 'python')

import caffe
import os

caffe.set_mode_cpu()

#model_def = caffe_root + 'examples/mnist/lenet_train_math_test_withoutBN.prototxt'
model_def = caffe_root + 'examples/mnist/lenet.prototxt'
#model_weights = caffe_root + 'examples/mnist/lenet_withoutBN_val_first_iter_1000.caffemodel'
model_weights = caffe_root + 'examples/mnist/lenet_original_caffemodel/lenet_original_iter_10000.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

# Input image
input_img = sys.argv[1]

#load the mean math image
#mu = np.load(caffe_root + 'examples/imagenet/math_first_mean.npy')
#mu = mu.mean(1).mean(1)
#print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))	# move image channels to outermost dimension
#transformer.set_mean('data', mu)		# subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 1) 		# rescale from[0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))	# swap channels from RGB to BGR

#set the size of the input
net.blobs['data'].reshape(1, 1, 28, 28)
#print "data", net.blobs['data'].shape
#fig=plt.figure()
#image=caffe.io.load_image(caffe_root + 'data/math/train_first/8/0000011.JPEG')
image=caffe.io.load_image(input_img)
#image=caffe.io.load.image(img)
ipdb.set_trace()
transformed_image=transformer.preprocess('data', image)
#img = Image.open(caffe_root + 'data/math/train_first/8/0000011.JPEG')
#img = Image.open(caffe_root + 'data/mnist/test/1-1.png')
#img.save('./test/test_image1digit_4_11.JPEG')
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

#perform classification
output = net.forward()

output_prob = output['prob'][0] # the output probabiility vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

#load labels
labels_file = caffe_root + 'data/math/label_1digit.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label:', labels[output_prob.argmax()]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5] #reverse sort and take 5 largest items
print top_inds
script=argv
target=open('./test/top5_mnist.txt','w')
print 'probabilities and labels:' 

print >>target, 'probabilities and labels:' 
#print >>target, zip(output_prob[top_inds], labels[top_inds]) 
print zip(output_prob[top_inds], labels[top_inds])
target.close()


# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
  print layer_name + '\t' + str(blob.data.shape)
for layer_name, param in net.params.iteritems():
  print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def vis_square(data):
  """Take an array of shape (n, height, width) or (n, height, width, 3)
     and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
  #normalize data for display
  data = (data - data.min()) / (data.max() - data.min())
  
  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]), 
             (0, 1), (0, 1)) + ((0,0),) * (data.ndim -3))
  data = np.pad(data, padding, mode='constant', constant_values=1)

  #tile the filters into an image
  plt.imshow(data); plt.axis('off')

##conv1
#filters = net.params['conv1'][0].data
#vis_square(filters.transpose(0, 2, 3, 1))
#fig.savefig('./test/conv1.JPEG')
#
##conv1_output
#feat = net.blobs['conv1'].data[0]
#vis_square(feat)
#plt.savefig('./test/conv1_output.JPEG')
#
##pool1
#filters = net.params['pool1'][0].data
#vis_square(filters.transpose(0, 2, 3, 1))
#plt.savefig('./test/pool1.JPEG')
#
##pool1_output
#feat = net.blobs['pool1'].data[0]
#vis_square(feat)
#plt.savefig('./test/pool1_output.JPEG')
#
##conv2
#filters = net.params['conv2'][0].data
#vis_square(filters.transpose(0, 2, 3, 1))
#plt.savefig('./test/conv2.JPEG')
#
##conv2_output
#feat = net.blobs['conv2'].data[0]
#vis_square(feat)
#plt.savefig('./test/conv2_output.JPEG')
#
##pool2
#filters = net.params['pool2'][0].data
#vis_square(filters.transpose(0, 2, 3, 1))
#plt.savefig('./test/pool2.JPEG')
#
##pool2_output
#feat = net.blobs['pool2'].data[0]
#vis_square(feat)
#plt.savefig('./test/pool2_output.JPEG')
#
##ip2
#feat = net.blobs['ip2'].data[0]
#plt.subplot(2, 1, 1)
#plt.plot(feat.flat)
#plt.subplot(2, 1, 2)
#_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
#plt.savefig('./test/ip2.JPEG')
#
## final probability output prob
#feat = net.blobs['prob'].data[0]
#plt.figure(figsize=(15,3))
#plt.plot(feat.flat)
#plt.savefig('./test/prob.JPEG')
