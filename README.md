# PCB_RCNN
PCB Missing Hole Detection using masked R-CNN
Read this it is Self Explanatory

# This is the Model Figure
![Model](https://github.com/RoKu1/PCB_RCNN/blob/master/Model.JPG)
# This is the Flow Chart Figure
![Flow Chart](https://github.com/RoKu1/PCB_RCNN/blob/master/Flow_Chart.JPG)

# Process
## 1 Masked RCNN using Tensorﬂow GPU
### 1.1 How to Install Mask RCNN for Keras
Object detection is a task in computer vision that involves identifying the presence, location, and type of one or more objects in a given image.It is a challenging problem that
involves building upon methods for object recognition (e.g. where are they), object localization (e.g. what are their extent), and object classifcation (e.g. what are they).
The Region-Based Convolutional Neural Network, or RCNN, is a family of convolutional
neural network models designed for object detection, developed by Ross Girshick, et al.
There are perhaps four main variations of the approach, resulting in the current pinnacle
called Mask RCNN. The Mask R-CNN introduced in the 2018 paper titled “Mask RCNN”
16is the most recent variation of the family of models and supports both object detection
and object segmentation. Object segmentation not only involves localizing objects in the
image but also specifes a mask for the image, indicating exactly which pixels in the image belong to the object.Mask RCNN is a sophisticated model to implement, especially as
compared to a simple or even state-of-the-art deep convolution neural network model. Instead of developing an implementation of the RCNN or Mask RCNN model from scratch,
we can use a reliable third-party implementation built on top of the Keras deep learning
framework.
####- Step 1: Clone the Mask R-CNN GitHub Repository. A repository is like a folder for
your project. Your project’s repository contains all of your project’s fles and stores each
fle’s revision history. You can also discuss and manage your project’s work within the
repository. ... Public repositories are visible to everyone. t can be local to a folder on
your computer, or it can be a storage space on GitHub or another online host. You can
keep code fles, text fles, image fles, you name it, inside a repository. This will create a
new local directory with the name Mask RCNN
####- Step 2: Install the Mask R-CNN Library. The library can be installed directly via
pip.
####- Step 3: Confrm the Library Was Installed. It is always a good idea to confrm that
the library was installed correctly. You can confrm that the library was installed correctly
by querying it via the pip command; for example: pip show mask-rcnn
### 1.2 How to Prepare a Dataset for Object Detection
The Mask RCNN is designed to learn to predict both bounding boxes for objects as well
as masks for those detected objects, and the MISSING HOLE dataset does not provide
masks. As such, we will use the dataset to learn a MISSING HOLE object detection task,
and ignore the masks and not focus on the image segmentation capabilities of the model.
There are a few steps required in order to prepare this dataset for modelling and we will
work through each in turn in this section, including downloading the dataset, parsing
the annotations fle, developing a MISSING HOLE Dataset object that can be used by
the Mask RCNN library, then testing the dataset object to confrm that we are loading
images and annotations correctly.
#### Step 1: Install Dataset The frst step is to download the dataset into your current
17working directory. This can be achieved by cloning the GitHub repository directly, as
follows: git clone https: github.com experiencor missing hole.git. This will create a new
directory called “missing hole” with a subdirectory called ‘images’ that contains all of the
JPEG photos of missing holes and a subdirectory called ‘annotes/‘ that contains all of
the XML fles that describe the locations of missing holes in each photoThis means that
we should focus on loading the list of actual fles in the directory rather than using a
numbering system.
#### Step 2: Parse Annotation File We can see that the annotation fle contains a “size”
element that describes the shape of the photograph, and one or more “object” elements
that describe the bounding boxes for the missing hole objects in the photograph. The
size and the bounding boxes are the minimum information that we require from each annotation fle. We could write some careful XML parsing code to process these annotation
fles, and that would be a good idea for a production system. Instead, we will short-cut
development and use XPath queries to directly extract the data that we need from each
fle, e.g. a size query to extract the size element and a object or a bndbox query to extract
the bounding box elements. Python provides the ElementTree API that can be used to
load and parse an XML fle and we can use the fnd() and fndall() functions to perform
the XPath queries on a loaded document.
#### Step 3: Develop MISSING HOLE Dataset Object The mask-rcnn library requires
that train, validation, and test datasets be managed by a mrcnn.utils.Dataset object.
This means that a new class must be defned that extends the mrcnn.utils.Dataset class
and defnes a function to load the dataset, with any name you like such as load dataset(),
and override two functions, one for loading a mask called load mask() and one for loading
an image reference (path or URL) called image reference(). To use a Dataset object, it
is instantiated, then your custom load function must be called, then fnally the built-in
prepare() function is called. The custom load function, e.g. load dataset() is responsible
for both defning the classes and for defning the images in the dataset. Classes are defned
by calling the built-in add class() function and specifying the ‘source‘ (the name of the
dataset), the ‘class id‘ or integer for the class (e.g. 1 for the frst lass as 0 is reserved
for the background class), and the ‘class name‘ such as missing hole. This will defne an
“image info” dictionary for the image that can be retrieved later via the index or order in
which the image was added to the dataset. You can also specify other arguments that will
18be added to the image info dictionary, such as an ‘annotation‘ to defne the annotation
path.
#### Step 4: Test MISSING HOLE Dataset Object The frst useful test is to confrm that
the images and masks can be loaded correctly. We can test this by creating a dataset
and loading an image via a call to the load image() function with an image id, then load
the mask for the image via a call to the load mask() function with the same image id.
Next, we can plot the photograph using the Matplotlib API, then plot the frst mask
over the top with an alpha value so that the photograph underneath can still be seen.
Running the example frst prints the shape of the photograph and mask NumPy arrays.
We can confrm that both arrays have the same width and height and only diﬀer in
terms of the number of channels. We can also see that the frst photograph (e.g. image
id=0). In this case only has one mask Finally, the mask-rcnn library provides utilities
for displaying images and masks. We can use some of these built-in functions to confrm
that the Dataset is operating correctly. For example, the mask-rcnn library provides the
mrcnn.visualize.display instances() function that will show a photograph with bounding
boxes, masks, and class labels. This requires that the bounding boxes are extracted from
the masks via the extract bboxes() function.
### 1.3 How to Train Mask RCNN Model for Missing Hole Detection
A Mask RCNN model can be ft from scratch, although like other computer vision applications, time can be saved and performance can be improved by using transfer learning.
The Mask RCNN model pre-ft on the MS COCO object detection dataset can be used as
a starting point and then tailored to the specifc dataset, in this case, the MISSING HOLE
dataset. The frst step is to download the model fle (architecture and weights) for the
pre-ft Mask RCNN model. The weights are available from the GitHub project and the
fle is about 250 megabytes. Next, a confguration object for the model must be defned.
This is a new class that extends the mrcnn.confg. Confg class and defnes properties of
both the prediction problem (such as name and the number of classes) and the algorithm
for training the model (such as the learning rate). The confguration must defne the
name of the confguration via the ‘NAME‘ attribute, e.g. ‘missing hole cfg‘, that will be
used to save details and models to fle during the run. The confguration must also defne
the number of classes in the prediction problem via the ‘NUM CLASSES‘ attribute. In
19this case, we only have one object type of hole, although there is always an additional
class for the background. Finally, we must defne the number of samples (photos) used
in each training epoch. This will be the number of photos in the training dataset.
### 1.4 How to Evaluate a Mask RCNN Model
The performance of a model for an object recognition task is often evaluated using the
mean absolute precision, or MAP. We are predicting bounding boxes so we can determine
whether a bounding box prediction is good or not based on how well the predicted and
actual bounding boxes overlap. This can be calculated by dividing the area of the overlap
by the total area of both bounding boxes, or the intersection divided by the union, referred
to as “intersection over union,” or IoU. A perfect bounding box prediction will have an IoU
of 1. It is standard to assume a positive prediction of a bounding box if the IoU is greater
than 0.5, e.g. they overlap by 50 percent or more. Precision refers to the percentage of
the correctly predicted bounding boxes (IoU > 0.5) out of all bounding boxes predicted.
Recall is the percentage of the correctly predicted bounding boxes (IoU > 0.5) out of all
objects in the photo. As we make more predictions, the recall percentage will increase,
but precision will drop or become erratic as we start making false positive predictions.
The recall (x) can be plotted against the precision (y) for each number of predictions to
create a curve or line. We can maximize the value of each point on this line and calculate
the average value of the precision or AP for each value of recall.
The average or mean of the average precision (AP) across all of the images in a
dataset is called the mean average precision, or MAP.
The Mask RCNN library provides a mrcnn.utils.compute ap to calculate the AP
and other metrics for a given images. These AP scores can be collected across a dataset
and the mean calculated to give an idea at how good the model is at detecting objects
in a dataset. First, we must defne a new Confg object to use for making predictions,
instead of training. We can extend our previously defned MissingholeConfg to reuse the
parameters. Instead, we will defne a new object with the same values to keep the code
compact. The confg must change some of the defaults around using the GPU for inference
that are diﬀerent from how they are set for training a model (regardless of whether you
are running on the GPU or CPU).
203.2.5 How to detect missing holes in new pictures
We can use the trained model to detect missing holes in new photographs, specifcally, in
photos that we expect to have missing holes. First, we need a new photo of a PCB. We
could go to Flickr and fnd a random photo of a PCB. Alternately, we can use any of the
photos in the test dataset that were not used to train the model. We have already seen in
the previous section how to make a prediction
### 1.5 How to detect missing holes in new pictures
We can use the trained model to detect missing holes in new photographs, specifcally, in
photos that we expect to have missing holes. First, we need a new photo of a PCB. We
could go to Flickr and fnd a random photo of a PCB. Alternately, we can use any of the
photos in the test dataset that were not used to train the model. We have already seen in
the previous section how to make a prediction with an image. Specifcally, scaling the pixel
values and calling model.detect(). Let’s take it one step further and make predictions for
a number of images in a dataset, then plot the photo with bounding boxes side-by-side
with the photo and the predicted bounding boxes. This will provide a visual guide to how
good the model is at making predictions. The frst step is to load the image and mask
from the dataset Next, we can make a prediction for the image. Next, we can create a
subplot for the ground truth and plot the image with the known bounding boxes. We can
then create a second subplot beside the frst and plot the frst, plot the photo again, and
this time draw the predicted bounding boxes in red. We can tie all of this together into a
function that takes a dataset, model, and confg and creates a plot of the frst fve photos
in the dataset with ground truth and predicted bound boxes.
