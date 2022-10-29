---
layout: post
title:  "Localization Learning Part 3"
tags:
  - deep learning
  - machine learning
  - tensorflow 2
  - Localization
  - tf.keras.utils.Sequence
  - transfer learning
  - vgg16
---

This is my implementation of the Localization part 3 lectures from Lazy Programmer's Deep Learning and Advanced Computer Vision course. The objective is to extend the code from Localization 1 and 2 and have it detect the bounding box of an image that can be scaled to different sizes.

Using a topless, pre-trained VGG16 network as a feature extractor we build a small dense layer with four outputs. The outputs represent the column, row, width and height of the Pokemon image.

As directed by the tutorial, I apply the sigmoid activation function to all four of the outputs.

Code for this article can be found at [https://github.com/jpmasters/jupyter-notebooks/blob/main/localization-3.ipynb](https://github.com/jpmasters/jupyter-notebooks/blob/main/localization-3.ipynb)


```python
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from PIL import Image, ImageDraw
```

I started with the final code from [Localization 2](https://github.com/jpmasters/jupyter-notebooks/blob/main/localization-2.ipynb) and then made the necessary modifications from there.


```python
IMAGE_DIM = 200
```

The LocalizationSequence class I created to load and generate the Pokemon images used the PIL library to load the image from disk. For this code, I need to be able to create the image at different scales. Fortunately, the PIL library includes a method for doing this on the Image class.


```python
with Image.open('assets/images/charmander.png') as im:
    im_small = im.resize((int(im.width * 0.5), int(im.height * 0.5)))
    im_medium = im.copy()
    im_large = im.resize((int(im.width * 1.5), int(im.height * 1.5)))
    
ax = plt.subplot(1, 3, 1)
ax.imshow(im_small)

ax = plt.subplot(1, 3, 2)
ax.imshow(im_medium)

ax = plt.subplot(1, 3, 3)
ax.imshow(im_large)

plt.show()
```


    
![Resized Pokemon images](/assets/localization-3-1.png)
    


I updated the LocalizationSequence to include the image resizing. The main difference between this and the code in Localization 2 is that, instead of converting the image into a numpy array as soon as it's loaded, I store a copy of the PIL image. Then in the `generate_charmander()` function, I can create a resized copy very easily and convert it to a numpy array later on. 


```python
class LocalizationSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        with Image.open('assets/images/charmander.png') as im:
            self.charmander = im.copy()
        
        
    def preprocess_y_value(self, y):
        """
        Sets the range of y to 0..1.
        """
        return y / IMAGE_DIM
        
        
    def generate_charmander(self):
        """
        Generates a Charmander pokemon of a random size from 0.5 to
        1.5 the scale of the original one.
        """
        scale = random.uniform(0.5, 1.5)
        poke = self.charmander.resize(
            (int(self.charmander.width * scale), int(self.charmander.height * scale))
        )
        
        return poke
        
        
    def generate_image(self):
        """
        Generates a random image with a Charmander inside it. 
        """
        img = np.zeros((IMAGE_DIM, IMAGE_DIM, 3))
        
        poke = self.generate_charmander()
        (char_width, char_height) = poke.size
        
        # create a numpy array from the image and strip out the
        # alpha channel
        poke_arr = np.array(poke)[:, :, 0:3]
        
        # normalize colours
        poke_arr = poke_arr / 255.
        
        char_top = random.randint(1, IMAGE_DIM - char_height)
        char_left = random.randint(1, IMAGE_DIM - char_width)
        
        img[char_top:char_top + char_height, 
            char_left:char_left + char_width] = poke_arr
        
        return img, self.preprocess_y_value(
            np.array([char_top, char_left, char_height, char_width])
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
        batch_x = np.empty(
            [self.batch_size, IMAGE_DIM, IMAGE_DIM, 3], 
            dtype=np.float32
        )
        batch_y = np.empty([self.batch_size, 4], dtype=np.float32)
        
        for i in range(self.batch_size):
            batch_x[i], batch_y[i] = self.generate_image()
            
        return batch_x, batch_y
    
```


Next, we can test out the LocalizationSequence by creating a batch of images. It doesn't matter what value is passed as the index parameter of `LocalizationSequence.__getitem__(idx)` as it'll generate a random batch every time.


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


    
![Sample output from generator](/assets/localization-3-2.png)
    



To recap, I created the model by starting with a trained, topless VGG16, flattening the outputs as they'll be 2D tensors, and the final dense layer with four outputs and sigmoid activation. After a bit of trial and error, I also added an extra dense layer which helped to bring the accuracy up a bit.

We want to make sure we don't try and train the Conv2D layers as it'll take too long so that part of the model is marked as untrainable.

It's then compiled with binary crossentropy as the loss function and the Adam optimizer. Note that the default learning rate (0.001) can be a bit too low so I reduced it.


```python
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
```


```python
# create the model
vgg16 = tf.keras.applications.vgg16.VGG16(
    input_shape=(IMAGE_DIM, IMAGE_DIM, 3), 
    include_top=False, 
    weights='imagenet'
)

# we don't want to train the VGG16 model
vgg16.trainable = False

# create the dense layer
x = tf.keras.layers.Flatten()(vgg16.output)
x = tf.keras.layers.Dense(500, activation="relu")(x)
x = tf.keras.layers.Dense(4, activation="sigmoid")(x)

# and build and compile it
model = tf.keras.Model(vgg16.input, x, name="Localization_Model")
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
```

    Model: "Localization_Model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_11 (InputLayer)       [(None, 200, 200, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 200, 200, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 200, 200, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 100, 100, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 100, 100, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 100, 100, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 50, 50, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 50, 50, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 50, 50, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 50, 50, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 25, 25, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 25, 25, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 25, 25, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 25, 25, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 12, 12, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 6, 6, 512)         0         
                                                                     
     flatten_10 (Flatten)        (None, 18432)             0         
                                                                     
     dense_13 (Dense)            (None, 500)               9216500   
                                                                     
     dense_14 (Dense)            (None, 4)                 2004      
                                                                     
    =================================================================
    Total params: 23,933,192
    Trainable params: 9,218,504
    Non-trainable params: 14,714,688
    _________________________________________________________________


You can see from the summary above that nearly all of the weights are untrainable but the final flatten and dense layer weights are.

Next we train the model using the LocalizationSequence object.


```python
history = model.fit(seq, epochs=5)
```

    Epoch 1/5
    32/32 [==============================] - 143s 4s/step - loss: 0.5872 - accuracy: 0.7969
    Epoch 2/5
    32/32 [==============================] - 165s 5s/step - loss: 0.5505 - accuracy: 0.9541
    Epoch 3/5
    32/32 [==============================] - 171s 5s/step - loss: 0.5520 - accuracy: 0.9648
    Epoch 4/5
    32/32 [==============================] - 170s 5s/step - loss: 0.5519 - accuracy: 0.9668
    Epoch 5/5
    32/32 [==============================] - 171s 5s/step - loss: 0.5501 - accuracy: 0.9619


Plot the accuracy from the `model.fit()`.


```python
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.show()
```


    
![Accuracy graph from training](/assets/localization-3-3.png)
    


Training can take some time so save out the model so it can be loaded from disk in future.


```python
model.save('saved-models/localization-3.h5', overwrite=True)
```

The `to_pil_rect()` function takes the y outputs from the model and converts the values into the correct coordinates for the input.


```python
def to_pil_rect(y):
    rv = y * IMAGE_DIM
    rv = np.clip(rv, 1, IMAGE_DIM - 1)
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

    Generated data for predictions: (32, 200, 200, 3)


Use the trained model to make some predictions from the new batch.


```python
y_predicted = model.predict(x)
```

    1/1 [==============================] - 4s 4s/step


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
    rv = np.zeros((seq.batch_size, IMAGE_DIM, IMAGE_DIM, 3), dtype=np.uint8)
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

Now we can draw some of the images that were used to create the predictions and overlay the localization rectangles on top of them as we did previously in Object Localization 1. 


```python
plt.rcParams["figure.figsize"] = (10,10)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    im_idx = random.randint(0, seq.batch_size - 1)
    ax.imshow(final_images[im_idx])
    ax.set_axis_off()
    ax.set_title(f'Image {im_idx}')
```


    
![Outp from model predictions](/assets/localization-3-4.png)
    


In the output above you can see the green boxes that are the results of the predictions and the yellow boxes that use the same maths to reverse the sigmoid function and validate that that maths is correct.
