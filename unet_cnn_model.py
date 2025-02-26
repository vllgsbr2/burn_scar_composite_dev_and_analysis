'''

Author            : Javier Villegas Bravo
Affiliation       : Univeristy of Maryland CISESS
Last Modified     : 07/02/2024
Code derived from :
   https://www.machinelearningnuggets.com/image-segmentation-with-u-net-define-u-net-model-from-scratch-in-keras-and-tensorflow/

Tensorflow Keras Documentation for all code: https://keras.io/api/

Goal : Build U-Net Convolutional Neural Network to classify burned and non burned
       land surface
Data :
     Input Data : daily burn scar composites RGB [2.25µm, 0.86µm, 0.67µm] BRF (capture spectral properties)
                  VZA, cos(SZA), RAA (capture sun-view geometry effects)
                  Day of year in 2 week blocks, block 0 to 25 (capture seasonal effects)

                Input Pre-processing : subset burn scars to get even ~50/50 split
                                       of data for all manually analyzed burns.
                                       Unburned land should have diverse land types
                                       including desert, salt beds, mtns,
                                       different vegetation types and ephemeral
                                       water bodies.
     Labelled Training Data : daily burn scar composites primitive burn scar mask (pbsm)
                     This mask is a good first try when intersected with the
                     manual analysis (refferring to when we subset the burn scars)

                     Eventually we need to move to the Monitoring Trends in Burn
                     Severity Dataset (MTBS).

                     https://www.mtbs.gov/viewer/index.html

                     This dataset is the gold standard
                     in burn scar mapping and will serve as the highest quality
                     ground truth to train our model. I just need to resample it
                     to our common grid before we use it. (Harder than it sounds...)

     Data Split : In machine learning and in any type of model evalualtion and
                  construction we must split our data pool into 3 categories,
                  1) Train 2) Validation 3) Test.

                  1) Training data is used to iteratively
                  update the values in the convolution filters (weaights and biases)
                  until they converge on values such that the model is in best
                  agreeance with the ground truth data. In our case the pbsm and
                  MTBS data. This data is also commonly reffered to as labelled
                  data cause it contains the true classification of each pixel.

                  2) Validation data a sample of data held back from training
                     your model that is used to give an estimate of model skill
                     while tuning model’s hyperparameters and holding
                     the weights/biases constant. Hyperparameters are the
                     things you can change about a model before during and after
                     training. Basically just the structure of the model. The
                     weights and biases are chosen by the training process itself
                     and can only be influnced by the hyperparamters during training.

                     For the U-Net CNN the hyperparamters are

                     a) The number and types of layers. Layers can include
                        convolution, normalization, max pooling, upsampling and
                        any combination of the above including any other modification
                        of the data. We could arbitrarily set negative values to
                        0 in the layer. In fact this is commonly done in CNNs.

                     b) learning rate. This is a factor which decides how much to
                        move in paramter space every traning epoch. Basically
                        during training, the model is changing the values in the
                        filters to map the input image of burned and unburned land
                        to an equal image but not the values are just 0 and 1 for
                        burned and unburned land which is the final classification.

                        The filters are weights and biases. The convolution
                        operation takes two arrays of the same size and multiplies
                        all them together so [[0,1], [2,3]] * [[0,1], [2,3]] would
                        equal [[0,1],[4,9]] and then it's all added together and replaces
                        the pixel over which the convolution is centered. So this
                        would be 0+1+4+9 = 13. Then there is a bias filter which just adds
                        a number to it. Lets say the bias filter is 1.

                        The take home message is output_layer_x = sum(w_i*x_i)+b
                        and that the operation of convolution is just carrying
                        out this operation. So really a CNN is giant network of
                        linear models working and learning together.

                     c) The number of filters, the filter size, and the convolution
                        stride and window size between the filter and the input
                        image. The filter holds numbers that are randomly intialized
                        when training the model. The model will then iteratively change these
                        values to minimize the error between the input and labelled data.
                        The filters only change value during training. After the
                        model is trained, these values stay constant.

                        The way these filters are used is illustrated here in it's
                        simplest form here
                        https://www.researchgate.net/figure/sualization-of-convolution-layer-26-Fig-1-demonstrates-how-the-filter-or-kernel-moves_fig1_351281913


                  3) Test data is withheld from the training and validation steps.
                     During model testing, hyperparameters cannot be changed. The
                     purpose of the test data set is to see if the model can generalize
                     well to new data in it's final form. It's also used to compare
                     different models to see how they do on the same dataset.

The model structure, training, monitoring and model output is discussed further
in the code

'''
import h5py
import tensorflow as tf
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

'''
Matrix X is NxLxWxF, where
N is the number of burn scars
L is the length of the image
W is the Width of the image
F is the number of features to train on i.e. reflectance, sun-view geometry etc...

Matrix y is NxLxW and just contains 1s and 0s where 1 is burned and 0 is unburned land

the data comes from the daily composite files that we have
'''

home_path = '/scratch/zt1/project/vllgsbr2-prj/burnscar_unet_cnn_data/training_data/'

x_path = home_path + 'input_data_X_matrix/cropped_burnscars_test.h5'
'''
X = []
with h5py.File(x_path, 'r') as x_composite: 
    rectangle_keys = list(x_composite.keys())
    
    for rectangle in rectangle_keys:
        X.append(x_composite[rectangle+"/data"][:])
'''
y_path = home_path + 'labelled_data_Y_matrix/cropped_burnscar_labels_test.h5'
'''
y = []
with h5py.File(y_path, 'r') as y_composite: 
    rectangle_keys = list(y_composite.keys())
    
    for rectangle in rectangle_keys:
        y_temp = y_composite[rectangle+"/data"][:]
        y_shape = np.shape(y_temp)     
        y_temp = y_temp.reshape((y_shape[0], y_shape[1], 1))
        y.append(y_temp)
'''

'''************************************************************************************'''

def load_data_with_padding(x_path, y_path):
    X = []
    with h5py.File(x_path, 'r') as x_composite:
        rectangle_keys = list(x_composite.keys())

        for rectangle in rectangle_keys:
            X.append(x_composite[rectangle+"/data"][:])


    y = []
    with h5py.File(y_path, 'r') as y_composite:
        rectangle_keys = list(y_composite.keys())

        for rectangle in rectangle_keys:
            y_temp = y_composite[rectangle+"/data"][:]
            y_shape = np.shape(y_temp)
            y_temp = y_temp.reshape(( y_shape[0], y_shape[1], 1))
            y.append(y_temp)

    max_height = 160 # max(img.shape[0] for img in X)
    max_width  = max(img.shape[1] for img in X)

    X_padded = [np.pad(img , ((0, max_height - img.shape[0]) , (0, max_width - img.shape[1]) , (0, 0)), mode='constant', constant_values=0) for img  in X]
    y_padded = [np.pad(mask, ((0, max_height - mask.shape[0]), (0, max_width - mask.shape[1]), (0, 0)), mode='constant', constant_values=0) for mask in y]

    return X_padded, y_padded, max_height, max_width

X, y, IMG_HEIGHT, IMG_WIDTH = load_data_with_padding(x_path, y_path)

'''************************************************************************************'''


#print(np.shape(X[0]))
#print(np.shape(y[0]))
#X = np.array(X)

#print(len(X))
#print(len(y))
#X=X[0].reshape((1,75,85,3))
#y=y[0].reshape((1,75,85,1))

#split train/val/test to 60%/20%/20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

'''
X_train = X_train.reshape(20,1,160,128,3) 
y_train = y_train.reshape(20,1,160,128,1)

X_val = X_val.reshape(20,1,160,128,3)
y_val = y_val.reshape(20,1,160,128,1)
'''

X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)

'''
a=X_train[0].shape[0]
b=X_train[0].shape[1]

c=X_val[0].shape[0]
d=X_val[0].shape[1]

e=X_test[0].shape[0]
f=X_test[0].shape[1]

X_train=X_train[0].reshape((1,a,b,3))
y_train=y_train[0].reshape((1,a,b,1))

X_val=X_val[0].reshape((1,c,d,3))
y_val=y_val[0].reshape((1,c,d,1))

X_test=X_test[0].reshape((1,e,f,3))
y_test=y_test[0].reshape((1,e,f,1))
'''

print(np.shape(X_train))
print(np.shape(y_train))

'''
Now define the graph of the model
This is where we define how the data will flow through the forward pass of the model
The forward pass is where the input data is turned into the final classification
and no training is done here. That is a seperate process called back propagation
which is a fancy word for taking the derivative, setting it to zero and solving for
x. That uses the structure of the graph to do the chain rule from the end of the
graph to the beggining (that's why it's called BACK propagation).
The math gets really messy but keras will take care of it when the training
happens later in the fit() function.
'''

num_classes = 1

#IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = None, None, 3
IMG_CHANNELS = 3

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#Contraction path
'''
Conv2D: (https://keras.io/api/layers/convolution_layers/convolution2d/)
takes the filter size, the window size, the activation function, the kernel
initializer and the padding (padding is the edge condition when the window is centered
over pixels in the edge of the input image. "same" padding just repeats the pixel,
"zero" padding just puts zeros).

the kernel_initializer is the technique used to randomly populate the filters.
bias filters can be explicitly initialized but by defualt they use 0s

Dropout: is a technique that randomly selects a percentage of the network, defined
by the graph we are building below, to ignore connections in the network. This
way certain connections aren't made more important than they should be. This is like
the way a human might learn X idea to be true, but it has lots of nuances depending
on different contexts.

BatchNormalization: Batch normalization applies a transformation that maintains
                    the mean output close to 0 and the output standard deviation
                    close to 1. It's is a method used to make training of
                    artificial neural networks faster and more stable through
                    normalization of the layers' inputs by re-centering and
                    re-scaling. It was proposed by Sergey Ioffe and Christian
                    Szegedy in 2015.

ReLu: just takes a convolved pixel, x, and does this f(x)=max(0,x). Which just makes the
      negative x into 0, otherwise nothing happens. Negative values can make the
      back propagtion go the wrong direction so instead of minimizing error it would
      maximize error. This step helps the CNN converge to the most optimal solution.

MaxPooling2D: using a window size (NxM) and stride (S) it take the pixels in the
              window and the max value replaces the NxM window and leaves 1 pixel.
              Then repeats for the whole image. If N=M=3 then the new image is 1/9
              the original amount of pixels. This technique highlights strongest
              signals in image while taking away the weak signals. When repeated
              over many layers, we see the different strong signals at all the
              scales from small to large.

Conv2DTranspose: https://stats.stackexchange.com/questions/252810/in-cnn-are-upsampling-and-transpose-convolution-the-same
                 Looks like we have a choice to upsample.
                 upsample2D and  Conv2DTranspose
                 upsample2D simply repeats pixels
                 Conv2DTranspose is a reverse convolution but still has many assumptions

                 The need for transposed convolutions generally arise from the
                 desire to use a transformation going in the opposite direction
                 of a normal convolution, i.e., from something that has the shape
                 of the output of some convolution to something that has the
                 shape of its input while maintaining a connectivity pattern
                 that is compatible with said convolution.

upsampling2D: The implementation uses interpolative resizing, given the resize
              method (specified by the interpolation argument). Use
              interpolation=nearest to repeat the rows and columns of the data.

concatenate: It takes as input a list of tensors, all of the same shape except
             for the concatenation axis, and returns a single tensor that is the
             concatenation of all inputs.

             This layer connects the downsmapling to the upampling section of the
             graph. This is what makes it a U-Net. If we didn't concatenate it
             would just be an encoder decoder. The U-Net is special because it
             forces the model to remember the information at smaller scales before
             the final classifcation. This has shown to improve semantic
             segmantation results. We may find it does not and can remove this,
             as we can remove and add any part to our model, during validation,
             or even retrain on another architechture as long as we improve results!


'''
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
b1 = tf.keras.layers.BatchNormalization()(c1)
r1 = tf.keras.layers.ReLU()(b1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)
print(tf.shape(p1), "p1 ************************************")

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
b2 = tf.keras.layers.BatchNormalization()(c2)
r2 = tf.keras.layers.ReLU()(b2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
print(tf.shape(p2), "p2*************************************")

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
b3 = tf.keras.layers.BatchNormalization()(c3)
r3 = tf.keras.layers.ReLU()(b3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
print(tf.shape(p3), "p3**************************************")

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
b4 = tf.keras.layers.BatchNormalization()(c4)
r4 = tf.keras.layers.ReLU()(b4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
print(tf.shape(p4), "p4**************************************")

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
b5 = tf.keras.layers.BatchNormalization()(c5)
r5 = tf.keras.layers.ReLU()(b5)
c5 = tf.keras.layers.Dropout(0.3)(r5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
print(tf.shape(c5), "c5***************************************")

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
u6 = tf.keras.layers.BatchNormalization()(u6)
u6 = tf.keras.layers.ReLU()(u6)
print(tf.shape(u6), "u6****************************************")

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
u7 = tf.keras.layers.concatenate([u7, c3])
u7 = tf.keras.layers.BatchNormalization()(u7)
u7 = tf.keras.layers.ReLU()(u7)
print(tf.shape(u7), "u7*****************************************")

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
u8 = tf.keras.layers.concatenate([u8, c2])
u8 = tf.keras.layers.BatchNormalization()(u8)
u8 = tf.keras.layers.ReLU()(u8)
print(tf.shape(u8), "u8*****************************************")

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
u9 = tf.keras.layers.BatchNormalization()(u9)
u9 = tf.keras.layers.ReLU()(u9)
print(tf.shape(u9), "u9******************************************")

outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)
print(tf.shape(outputs), "outputs*********************************")

'''
Initialize the model by feeding it the graph and the inputs.
The graph has no data in it. It's just a big long mathematical function we assigned
to the varibale called output.

The input is part of the graph but also has no data in it. It's just constraining
what can be input into the graph.

In object oriented programming, we initialize objects defined in a class.
For example, here we define an object from the Model class as defined by the
Tensflow Keras package we imported. This object when initialized needs to know
the graph by way of the input and output variables we defined.

to make the distinction from the class "Model", when we intialize we can call it
something else, like "unet_cnn_model". It's completely arbitrary what we name it.
Just like any other variable.
'''
unet_cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

'''
compile: Configures the model for training.

The optimizer chosen here is adam. This is a hyperparamter we could change. This
is how the model will perform back propagation. In general this is done by looking
a small amount (the learning rate) in every direction of the loss function in
parameter space and changing the weights and biasies such that they move in the
direction which minimizes loss. We are looking for a global minimum in loss (error).
We don't know the answer to every part of the loss function. But if we have enough
training data and train for enough epochs, then we have a good chance of finding it.
This is the main point of machine learning actually.

The loss function here is binary_crossentropy. There are many to choose.

metrics is the style of output we want. Here it is accuracy. Keras has many
metrics to look at to see how the model is doing during training. We may care
about lots of different metrics depending on what we want the model to do.
This will tell us if the model is actually converging on good weights/biases.
Otherwise we need to re-evaluate.

'''
unet_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

'''
# this line plots what the graph of the model looks like. Good for paper and
# model structure change over time as we tweak the hyperparamters.
tf.keras.utils.plot_model(unet_cnn_model, "model.png")
'''
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir='logs')]

# batch size is the number of burn scars to analyze before changing the weights/biases.
# epochs is the number of times to train each batch.
# callbacks helps us monitor the model during training

unet_cnn_model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=20, epochs=100, callbacks=callbacks)

'''
for XX_train, yy_train, XX_val, yy_val in zip(X_train, y_train, X_val,y_val):
    XX_train, yy_train, XX_val, yy_val = XX_train.reshape(1,160,128,3), yy_train.reshape(1,160,128,1), XX_val.reshape(1,160,128,3), yy_val.reshape(1,160,128,1)
    unet_cnn_model.fit(XX_train, yy_train, validation_data=(XX_val,yy_val), batch_size=16, epochs=100)#, callbacks=callbacks)
'''
'''
# plot some stuff to monitor the training

loss = unet_cnn_model.history.history['loss']
val_loss = unet_cnn_model.history.history['val_loss']

plt.figure()
plt.plot( loss, 'r', label='Training loss')
plt.plot( val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
'''

