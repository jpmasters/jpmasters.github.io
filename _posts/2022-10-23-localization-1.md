---
layout: post
title:  "Localization Learning Part 1"
tags:
  - deep learning
  - machine learning
  - tensorflow 2
  - Localization
  - tf.keras.utils.Sequence
---

This is my implementation of the Localization part 1 lectures from Lazy Programmer's Deep Learning and Advanced Computer Vision course. The objective is to start by creating black and white images where there is a black background and a single randomly placed, random sized white rectangle. We then train a network to output the location of the rectangle.

Using a topless, pre-trained VGG16 network as a feature extractor we build a small dense layer with four outputs. The outputs represent the column, row, width and height of the white rectangle.

As directed by the tutorial, I apply the sigmoid activation function to all four of the outputs.


```python
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from PIL import Image, ImageDraw
```

Create the model by starting with a trained, topless VGG16, flattening the outputs as they'll be 2D tensors, and the final dense layer with four outputs and sigmoid activation.

We want to make sure we don't try and train the Conv2D layers as it'll take too long so that part of the model is marked as untrainable.

It's then compiled with binary crossentropy as the loss function and the Adam optimizer.


```python
# create the model
vgg16 = tf.keras.applications.vgg16.VGG16(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)

# we don't want to train the VGG16 model
vgg16.trainable = False

# create the dense layer
x = tf.keras.layers.Flatten()(vgg16.output)
x = tf.keras.layers.Dense(4, activation="sigmoid")(x)

# and build and compile it
model = tf.keras.Model(vgg16.input, x, name="Localization_Model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

```

    Model: "Localization_Model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_3 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
     flatten_2 (Flatten)         (None, 25088)             0         
                                                                     
     dense_2 (Dense)             (None, 4)                 100356    
                                                                     
    =================================================================
    Total params: 14,815,044
    Trainable params: 100,356
    Non-trainable params: 14,714,688
    _________________________________________________________________


You can see from the summary above that nearly all of the weights are untrainable but the final flatten and dense layer weights are.

Next we create a generator for the training images. It works by creating them at random on the fly. For this I created class derived from [tf.keras.utils.Sequence](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence).

Note that while the sigmoid function is good with values of x from -5 to 5 (and probably beyond), I got better results by limiting the range to -1 to 1. This might mean it'll work better with a different activation function (e.g. identity or leaky ReLU) but I didn't try that this time.


```python
class LocalizationSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.size_rows = 224
        self.size_cols = 224
        
        
    def preprocess_y_value(self, y):
        """
        Sets the range of y to -1..1 and applies the sigmoid function
        so it matches the output of the model.
        """
        rv = (y / (224 / 2))
        rv = rv - 1
        return tf.math.sigmoid(rv)
        
        
    def generate_image(self):
        """
        Generates a random image with a black background and a single white
        rectangle inside it. 
        """
        img = np.zeros((self.size_rows, self.size_cols, 3))
        block_top, block_left = random.randint(1, 224 - 2), random.randint(1, 224 - 2)
        block_width = random.randint(1, 224 - 1 - block_left)
        block_height = random.randint(1, 224 - 1 - block_top)
        
        for row in range(block_top, block_top + block_height):
            for col in range(block_left, block_left + block_width):
                img[row][col] = [1., 1., 1.]
        
        return img, self.preprocess_y_value(
            np.array([block_top, block_left, block_height, block_width])
        )
        
        
    def __len__(self):
        """
        We generate the images on the fly so just return the batch size.
        """
        return self.batch_size
    
    
    def __getitem__(self, idx):
        """
        Return a batch of images preprepared for training. I.e. colour ranges
        are 0..1 and not 0..255.
        """
        batch_x = np.empty([self.batch_size, 224, 224, 3], dtype=np.float32)
        batch_y = np.empty([self.batch_size, 4], dtype=np.float32)
        
        for i in range(self.batch_size):
            batch_x[i], batch_y[i] = self.generate_image()
            
        return batch_x, batch_y
            
```

Test out the LocalizationSequence by creating a batch of images. It doesn't matter what value is passed as the index parameter of `LocalizationSequence.__getitem__(idx)` as it'll generate a random batch every time.


```python
seq = LocalizationSequence(batch_size=32)
batch_x, batch_y = seq.__getitem__(0)
```

Now we can draw them out and see what we generated.


```python
plt.rcParams["figure.figsize"] = (10, 10)
for i in range(min(50, seq.__len__())):
    ax = plt.subplot(10, 5, i + 1)
    ax.set_axis_off()
    ax.imshow(batch_x[i])
    
plt.show()
```


    
![Sample output from the generator](/assets/localization-1-1.png)
    


Next we train the model using the LocalizationSequence object.


```python
history = model.fit(seq, epochs=10)
```

    Epoch 1/10
    32/32 [==============================] - 174s 5s/step - loss: 0.7119 - accuracy: 0.5967
    Epoch 2/10
    32/32 [==============================] - 184s 6s/step - loss: 0.6551 - accuracy: 0.8496
    Epoch 3/10
    32/32 [==============================] - 195s 6s/step - loss: 0.6513 - accuracy: 0.9033
    Epoch 4/10
    32/32 [==============================] - 206s 6s/step - loss: 0.6499 - accuracy: 0.9395
    Epoch 5/10
    32/32 [==============================] - 204s 6s/step - loss: 0.6496 - accuracy: 0.9199
    Epoch 6/10
    32/32 [==============================] - 190s 6s/step - loss: 0.6508 - accuracy: 0.9434
    Epoch 7/10
    32/32 [==============================] - 191s 6s/step - loss: 0.6504 - accuracy: 0.9521
    Epoch 8/10
    32/32 [==============================] - 193s 6s/step - loss: 0.6505 - accuracy: 0.9336
    Epoch 9/10
    32/32 [==============================] - 195s 6s/step - loss: 0.6499 - accuracy: 0.9336
    Epoch 10/10
    32/32 [==============================] - 196s 6s/step - loss: 0.6490 - accuracy: 0.9346


Plot the accuracy from the `model.fit()`.


```python
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.show()
```


    
![Training accuracy graph](/assets/localization-1-2.png)
    


Training can take some time so save out the model so it can be loaded from disk in future.


```python
model.save('saved-models/localization-1.h5', overwrite=True)
```

The `reverse_sigmoid()` function reverses the sigmoid activation to a value proportional to the image size.


```python
def reverse_sigmoid(y):
    return np.log(y / (1 - y))
```

The `to_pil_rect()` function takes the y outputs from the model, reverses the sigmoid activation and converts the values into the correct coordinates for the input.


```python
def to_pil_rect(y):
    rv = reverse_sigmoid(y)
    rv = (rv + 1) * 112
    rv = np.clip(rv, 1, 223)
    [top, left, height, width] = rv
    height += top
    width += left
    rv[2] = height
    rv[3] = width
    return rv
```

The sequence object can be used to create a set of images we can use for prediction.


```python
x, y = seq.__getitem__(0)
print(f'Generated data for predictions: {x.shape}')
```

    Generated data for predictions: (32, 224, 224, 3)


As you can see, we've created a batch of 32 test images.


```python
x.shape
```




    (32, 224, 224, 3)



Use the trained model to make some predictions from the new batch.


```python
y_predicted = model.predict(x)
```

    1/1 [==============================] - 6s 6s/step


Create the boxes for both the actual boxes and the predicted ones.


```python
predicted_boxes = np.zeros((seq.batch_size, 4))
actual_boxes = np.zeros((seq.batch_size, 4))

for i, pred in enumerate(y_predicted):
    predicted_boxes[i] = to_pil_rect(pred)
    actual_boxes[i] = to_pil_rect(y[i])
```

Convert the numpy array into a PIL image so we can work with it as an image.


```python
def add_bounding_boxes():
    rv = np.zeros((seq.batch_size, 224, 224, 3), dtype=np.uint8)
    for i in range(seq.batch_size):
        im = Image.fromarray(np.uint8(x[i] * 255), mode='RGB')

        draw = ImageDraw.Draw(im)

        draw.rectangle(
            [actual_boxes[i][1], 
             actual_boxes[i][0], 
             actual_boxes[i][3], 
             actual_boxes[i][2]], outline='yellow')

        draw.rectangle(
            [predicted_boxes[i][1], 
             predicted_boxes[i][0], 
             predicted_boxes[i][3], 
             predicted_boxes[i][2]], outline='green')
        
        rv[i] = np.array(im).astype(dtype=np.uint8)

    return rv
```


```python
final_images = add_bounding_boxes()
```

Now we can draw some of the images that were used to create the predictions and overlay the localization rectangles on top of them. You can see that the accuracy isn't brilliant but it's not too bad considering it's a very simple dense network and the feature extraction part of the VGG model probably isn't designed for images this simple. In future experiments, this will change to something more interesting.


```python
plt.rcParams["figure.figsize"] = (10,10)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    im_idx = random.randint(0, seq.batch_size - 1)
    ax.imshow(final_images[im_idx])
    ax.set_axis_off()
    ax.set_title(f'Image {im_idx}')
```


    
![Final output](/assets/localization-1-3.png)
    


In the output above you can see the green boxes that are the results of the predictions and the yellow boxes that use the same maths to reverse the sigmoid function and validate that that maths is correct.

## Further Experiments
1. I'd like to remove the sigmoid activation function as I'm not sure this is really adding anything. Perhaps try some of the ones found [here](https://en.wikipedia.org/wiki/Activation_function).
2. Dynamically change the learning rate during `model.fit()` to see if I can get some more accuracy out of the model.
