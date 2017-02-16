import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['GLOG_minloglevel'] = '2' 

# Make sure that caffe is on the python path:
#caffe_root = '/Users/anh/src/caffe_latest/'  # this file is expected to be in {caffe_root}/examples
caffe_root = '/Users/thaophung/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import ipdb

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = '/Users/anh/src/caffe_latest/examples/mnist/lenet_math_overfit.prototxt'
#MODEL_FILE = '/Users/thaophung/caffe/python/for_thao/lenet_math_overfit.prototxt'
MODEL_FILE = '/Users/thaophung/caffe/examples/math/lenet/lenet.prototxt'
#PRETRAINED = '/Users/anh/src/caffe_latest/examples/mnist/lenet_overfit_iter_500.caffemodel'
#PRETRAINED = '/Users/thaophung/caffe/python/for_thao/lenet_overfit_iter_500.caffemodel'
PRETRAINED = '/Users/thaophung/caffe/examples/math/caffemodel/lenet_math_iter_2000.caffemodel'
IMAGE_FILE = sys.argv[1] #'/Users/anh/workspace/cnn_math/train_10_image/0/0000017.JPEG'

def preprocess(img):
    img = np.float32(np.rollaxis(img, 2)[::-1])
    img = img[np.newaxis]
    return img

# Create a network
net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(72, 32))

input_image = caffe.io.load_image(IMAGE_FILE)
input_image = preprocess(input_image)

print input_image.mean(), input_image.max(), input_image.shape
#ipdb.set_trace()

# Put in the data layer
net.blobs['data'].data[...] = input_image
net.forward()
prediction = net.blobs['prob'].data.copy()
#    print 'prediction shape:', prediction[0].shape
# plt.plot(prediction[0])
predicted_class = prediction[0].argmax()
print 'predicted class:', predicted_class
print 'highest prob:', prediction[0][predicted_class] 
