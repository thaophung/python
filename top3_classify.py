import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
import caffe
import textwrap
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid

# Don't use Type 3 fonts when saving pdf
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'path'

caffe_root = '/Users/thaophung/caffe/' 	# this file is expected to be in {caffe_root}/examples

# Set the right path to your model definition file, pretrained model weights, 
# and the image you would like to classify. 
MODEL_FILE = '/Users/thaophung/caffe/examples/mnist/math_deploy.prototxt'
#MODEL_FILE = '/Users/thaophung/caffe/examples/mnist/lenet.prototxt'
#MODEL_FILE = '/Users/thaophung/caffe/models/bvlc_reference_caffenet/deploy.prototxt'

#PRETRAINED = '/Users/thaophung/caffe/examples/mnist/lenet_original_caffemodel/lenet_original_iter_10000.caffemodel'
PRETRAINED = '/Users/thaophung/caffe/examples/mnist/lenet_10_image_iter_250.caffemodel'
#PRETRAINED = '/Users/thaophung/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

def classifyImage(file_path):
  net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims = (72, 32))
  
  input_image = caffe.io.load_image(file_path) 
  input_image = np.float32(np.rollaxis(input_image, 2)[::-1])
  input_image = input_image[np.newaxis]
  prediction = net.predict([input_image])[0] 	# predict takes any number of images, and formats them for the Caffe net automatically

  return input_image, prediction

# Probability histogram 
# plt.plot(prediction[0])
# plt.show()

#------------------------------

def drawImage(input_image, prediction):
  #plt.subplot(211)
  ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)

  # Now print out the image and the probability 
  plt.imshow(input_image)
  xticks([])
  yticks([])

#---------------------------
def getPredictions(prediction):
  #f = open(caffe_root + "/data/ilsvrc12/synset_words.txt", 'r')
  f = open(caffe_root + "/data/math/label_1digit.txt", 'r')
  out = f.readlines() 	# will append in the list out

  probabilities = []
  i = 0
  
  for line in out:
    line = line.strip('\n')
   
    words = line.split(' ', 1)
    cat_id = words[0]
    cat_desc = words[1].split(',', 1)[0]
    
    #probabilities.append((prediction[i], cat_id, cat_desc))
  
    # Only probability followed by description of category
    probabilities.append((prediction[i], cat_desc))
    i = i + 1
 
  sorted_probabilities = sorted(probabilities, key=lambda tup: tup[0], reverse=True)
  
  val = []
  labels = []
  for index in range(0,5):
    # print sorted_probabilities[index]
    val.append(sorted_probabilities[4-index][0])
    labels.append(sorted_probabilities[4-index][1])
  return (val[2:5], labels[2:5])

#---------------------------------
def drawPredictions(values, labels):
  #ax = plt.subplot(212)
  ax = plt.subplot2grid((3,1), (2,0))
  
  pos = arange(4) + 0.5	# the bar centers on the y axis
  
  rects = barh(pos, values, height=1, align='center', color = "#a6a6ff", edgecolor = '#7979ba')
  ax.set_xlim([0, 1])

  xticks([0, 0.25, 0.5, 0.75, 1])
  # ax.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
  ax.set_xticklabels([])
  
  yticks([])

  #ax.spines['top'].set_visible(False)
  #ax.spines['bottom'].set_visible(False)
  #ax.get_xaxis().tick_bottom()
  ax.get_xaxis().set_visible(False)

  #Lastly, write in the ranking inside each bar to aid in interpretation
  for rid, rect in enumerate(rects):
    # Rectangle widths are already integer-valued but are floating
    # type, so it helps to remove the trailing decimal point and 0 by
    # converting width to int type
    width = int(rect.get_width())
   
    # Figure out what the last digit (width modulo 10) so we can add
    # the appropriate numerical suffix (e.g., 1st, 2nd, 3rd, etc)
    # lastDigit = width % 10
    # Note that 11, 12, and 13 are special cases
    # if (width == 11) or (width == 12) or (width == 13):
        # suffix = 'th'
    # else:
        #suffix = suffixes[lastDigit]
    # rankStr = str(width)
    if (width < 5):		# The bars aren't wide enough to print the ranking inside 
      xloc = width + 0.01	# Shift the text to the right side of the right edge
      clr = 'black'		# Black against white background
      align = 'left'
    else:
      xloc = 0.98*width 	# Shift the text to the left side of the right edge
      clr = 'white'		# White on magenta
      align = 'right'

    width = 0
    xloc = width + 0.95		# Shift the text to the right side of the right edge
    clr = '#111111'		# Black against white background
    align = 'right'

    # Wrap the label down to the next line
    w = labels[rid].split(",")
    shorten_label = w[0]

    '''
    if len(w) > 1:
      shorten_label += ", " + w[1]
    '''
    
    wrapped_label = '\n'.join(textwrap.wrap(shorten_label,25))
  
    # Center the text vertically in the bar
    yloc = rect.get_y() + rect.get_height()/2.0
    #text(xloc, yloc, wrapped_label, horizontalalignment=align, 
		#verticalalignment='center', color=clr, weight='normal', size='13',
		#path_effects=[PathEffects.withStroke(linewidth=3, foreground="w", alpha=1)])

    fontstyle = 'normal'
  
    #rect.set_color("#a6a6ff")

    # Human prediction label
    if (rid == 3):
      # rect.set_alpha(0.0)
      rect.set_color("white")
      xloc = 0.5
      align = 'center'
  
      rect.set_edgecolor("black")
      #rect.set_edgecolor((1.0, 0, 0, 0.0))
      #fontstyle='italic'

    # No stroke
    text(xloc, yloc, wrapped_label, horizontalalignment=align, 
		verticalalignment='center', color=clr, fontstyle=fontstyle, weight='normal', size='14')

#-------------------------------------------
def generatePlot(path_dir, file_name):
  path_file = path_dir + "/" + file_name 	# Image to classify
  path_save_file = path_dir + "/results/" + file_name + ".svg"
  
  image, prediction = classifyImage(path_file)

  #figure(1)
  fig = plt.figure(figsize=(4.5, 7.5), facecolor='w')
  #plt.subplot_adjust(hspace=0.,wspace=0.)

  #fig = plt.figure()
  grid = AxesGrid(fig, 111, # similar to subplot(141)
 			nrows_ncols = (2,1),
			axes_pad = 0.0, 
			label_mode = "1")
  values, labels = getPredictions(prediction)

  # Add human predictions
  human_label = file_name.replace("_", " ").replace(".jpg", "")
  values.append(1.0)
  labels.append(human_label)

  drawPredictions(values, labels)
  
  drawImage(image, prediction)
 
  fig.subplots_adjust(left=0.01, bottom=0.501, top=0.95, right=0.5, wspace = 0.05, hspace=0.05)

  #print values
  
  #plt.axis('equal')
  #plt.tight_layout()

  #Show  pop-up image
  #plt.show()

  # Save as image instead of pop-up
  plt.savefig(path_save_file, bbox_inches='tight', dpi=150, pad_inches=0)
  plt.close()

  return None

# ---- Main function ----
#path = "images/hen_white_bg.jpg"

if __name__ == '__main__':
  # print "TESTING"
  # Path to the directory containing images
  #path_dir = "/Users/thaophung/caffe/examples/images"
  path_dir = "/Users/thaophung/caffe/data/math/test_10_image"

  #ACCURACY = 0

  import os
  for file in os.listdir(path_dir):
    if file.endswith(".jpg"):
      path = path_dir + "/" + file
      print path
      #a = generatePlot(path_dir, file)
      generatePlot(path_dir, file)

      #if a == "CPPN images":
         #ACCURACY += 1
  #print "Accuracy: ", ACCURACY
