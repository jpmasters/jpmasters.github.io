---
layout: post
title:  "Localization Learning Part 7"
tags:
  - deep learning
  - machine learning
  - tensorflow 2
  - Localization
  - tf.keras.utils.Sequence
  - transfer learning
  - vgg16
  - tf.keras.layers
  - tf.keras.layers.Concatenate
---


This is my implementation of the Localization part 7 lectures from Lazy Programmer's Deep Learning and Advanced Computer Vision course. In this section we train the model to distinguish between three different Pokemon as well as detect whether a Pokemon exists in the scene and where it is.

We do this by adding an additional 3 outputs to the model that use the softmax activation function, and classify the Pokemon type if one is present in the scene.

Code for this article can be found at [https://github.com/jpmasters/jupyter-notebooks/blob/main/localization-7.ipynb](https://github.com/jpmasters/jupyter-notebooks/blob/main/localization-7.ipynb)


```python
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
from PIL import Image, ImageDraw

IMAGE_DIM = 200
```

I started with the final code from [Localization 6](https://github.com/jpmasters/jupyter-notebooks/blob/main/localization-6.ipynb) and then made the necessary modifications from there using the same background photo by [Patrick Szylar](https://unsplash.com/@patrick_szylar?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/outdoor?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)


To combine the Pokemon and background images together, as before, I used the [`Image.alpha_composite()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.alpha_composite) function from Pillow which ensures that the transparency is respected.


In order to randomly generate some images with a Pokemon and some without, I updated the `generate_image()` function so that it only generates an image with a Pokemon in it 50% of the time.


```python
class LocalizationSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        with Image.open('assets/images/charmander.png') as ch, \
             Image.open('assets/images/bulbasaur.png') as bu, \
             Image.open('assets/images/squirtle.png') as sq, \
             Image.open('assets/images/landscape.png') as bg:
                
            self.pokemon = [ch.copy(), bu.copy(), sq.copy()]
            self.background = bg.copy()
        
        
    def preprocess_y_value(self, y):
        """
        Sets the range of y to 0..1.
        """
        return y / IMAGE_DIM
        

    def generate_background(self):
        """
        Generates a background image by cropping out a part of the
        background image.
        """
        left = random.randint(0, self.background.width - IMAGE_DIM)
        top = random.randint(0, self.background.height - IMAGE_DIM)
        return self.background.crop((left, top, left + IMAGE_DIM, top + IMAGE_DIM))

        
    def generate_pokemon(self, i):
        """
        Generates a Charmander pokemon of a random size from 0.5 to
        1.5 the scale of the original one and flipped left to right
        50% of the time.
        """
        scale = random.uniform(0.5, 1.5)
        poke = self.pokemon[i].resize(
            (int(self.pokemon[i].width * scale), 
             int(self.pokemon[i].height * scale))
        )

        if random.random() > 0.5:
            poke = poke.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            
        return poke
        
        
    def generate_image(self):
        """
        Generates a random image with a Charmander inside it. 
        """
        
        # start by generating the background
        img = self.generate_background()

        # some default values i case we don't end up generating a pokemon
        char_width, char_height, char_left, char_top = 0, 0, 0, 0

        # decide if to generate a pokemon and if so, which one
        poke_idx = random.randint(0, len(self.pokemon))
        
        include_pokemon = 1. if poke_idx < len(self.pokemon) else 0.
        
        # if poke_idx is 4 then don't generate a pokemon otherwise
        # poke_idx is the index of the pokemon in the pokemon array
        if include_pokemon ==1.:
            
            # generate the pokemon
            poke = self.generate_pokemon(poke_idx)
            (char_width, char_height) = poke.size

            # place the charminder randomly in the image
            char_left = random.randint(0, img.width - char_width)
            char_top = random.randint(0, img.height - char_height)

            # combine the two together
            img.alpha_composite(poke, dest=(char_left, char_top))
        
        # create a numpy array from the image and strip out the
        # alpha channel
        img_arr = np.array(img)[:, :, 0:3]
        
        # normalize colours
        img_arr = img_arr / 255.
        
        # normalise the values relating to dimensions
        dims = np.array([char_top, char_left, char_height, char_width]) / IMAGE_DIM
        
        # create three values to indicate which pokemon we're showing
        poke = np.zeros((4,))
        if include_pokemon == 1.:
            poke[poke_idx] = 1.
            poke[-1] = 1.
        
        return img_arr, np.append(dims, poke)
        
        
    def __len__(self):
        """
        We generate the images on the fly so just return a high number.
        """
        return 1000
    
    
    def __getitem__(self, idx):
        """
        Return a batch of images preprepared for training. I.e. colour ranges
        are 0..1 and not 0..255.
        """
        batch_x = np.empty(
            [self.batch_size, IMAGE_DIM, IMAGE_DIM, 3], 
            dtype=np.float32
        )
        batch_y = np.empty([self.batch_size, 8], dtype=np.float32)
        
        for i in range(self.batch_size):
            batch_x[i], batch_y[i] = self.generate_image()
            
        return batch_x, batch_y
    
```


Let's test that out and see what it generates.


```python
pokemon_categories = ["Charminder", "Bulbasaur", "Squirtle"]
seq = LocalizationSequence(batch_size=32)
batch_x, batch_y = seq.__getitem__(0)

plt.rcParams["figure.figsize"] = (10, 10)
for i in range(min(50, seq.batch_size)):
    ax = plt.subplot(10, 5, i + 1)
    ax.set_axis_off()
    ax.imshow(batch_x[i])
    
plt.show()
```


    
![Generated training images](/assets/localization-7-1.png)
    


Next, have a quick check of the Y data to make sure it's generated the right thing.


```python
batch_y
```




    array([[0.64 , 0.485, 0.305, 0.23 , 1.   , 0.   , 0.   , 1.   ],
           [0.565, 0.07 , 0.325, 0.345, 0.   , 1.   , 0.   , 1.   ],
           [0.475, 0.055, 0.43 , 0.415, 0.   , 0.   , 1.   , 1.   ],
           [0.48 , 0.075, 0.365, 0.35 , 0.   , 0.   , 1.   , 1.   ],
           [0.45 , 0.16 , 0.415, 0.435, 0.   , 1.   , 0.   , 1.   ],
           [0.35 , 0.06 , 0.415, 0.405, 0.   , 0.   , 1.   , 1.   ],
           [0.19 , 0.255, 0.23 , 0.175, 1.   , 0.   , 0.   , 1.   ],
           [0.185, 0.475, 0.275, 0.29 , 0.   , 1.   , 0.   , 1.   ],
           [0.17 , 0.09 , 0.365, 0.38 , 0.   , 1.   , 0.   , 1.   ],
           [0.6  , 0.05 , 0.15 , 0.115, 1.   , 0.   , 0.   , 1.   ],
           [0.085, 0.42 , 0.31 , 0.295, 0.   , 0.   , 1.   , 1.   ],
           [0.04 , 0.17 , 0.345, 0.36 , 0.   , 1.   , 0.   , 1.   ],
           [0.305, 0.53 , 0.315, 0.335, 0.   , 1.   , 0.   , 1.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.28 , 0.325, 0.39 , 0.38 , 0.   , 0.   , 1.   , 1.   ],
           [0.77 , 0.515, 0.23 , 0.175, 1.   , 0.   , 0.   , 1.   ],
           [0.185, 0.15 , 0.325, 0.34 , 0.   , 1.   , 0.   , 1.   ],
           [0.6  , 0.175, 0.175, 0.13 , 1.   , 0.   , 0.   , 1.   ],
           [0.085, 0.555, 0.4  , 0.39 , 0.   , 0.   , 1.   , 1.   ],
           [0.47 , 0.655, 0.335, 0.255, 1.   , 0.   , 0.   , 1.   ],
           [0.695, 0.725, 0.15 , 0.155, 0.   , 1.   , 0.   , 1.   ],
           [0.285, 0.555, 0.35 , 0.265, 1.   , 0.   , 0.   , 1.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.32 , 0.49 , 0.23 , 0.22 , 0.   , 0.   , 1.   , 1.   ],
           [0.35 , 0.33 , 0.155, 0.165, 0.   , 1.   , 0.   , 1.   ],
           [0.1  , 0.625, 0.36 , 0.35 , 0.   , 0.   , 1.   , 1.   ],
           [0.2  , 0.58 , 0.36 , 0.345, 0.   , 0.   , 1.   , 1.   ],
           [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.59 , 0.645, 0.265, 0.255, 0.   , 0.   , 1.   , 1.   ],
           [0.36 , 0.61 , 0.255, 0.195, 1.   , 0.   , 0.   , 1.   ],
           [0.695, 0.695, 0.25 , 0.26 , 0.   , 1.   , 0.   , 1.   ]],
          dtype=float32)



To read the 'pokemon generated / detected' data, we can slice out the last value in each row.


```python
batch_y[:,-1]
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1.,
           1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1.],
          dtype=float32)



The Pokemon character class data comes from the values at indexes 4, 5 and 6.


```python
batch_y[:,4:7]
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 1.],
           [0., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]], dtype=float32)



As previously, the location, width and height data is in the first 4 values of each array item.


```python
batch_y[:,0:4]
```




    array([[0.64 , 0.485, 0.305, 0.23 ],
           [0.565, 0.07 , 0.325, 0.345],
           [0.475, 0.055, 0.43 , 0.415],
           [0.48 , 0.075, 0.365, 0.35 ],
           [0.45 , 0.16 , 0.415, 0.435],
           [0.35 , 0.06 , 0.415, 0.405],
           [0.19 , 0.255, 0.23 , 0.175],
           [0.185, 0.475, 0.275, 0.29 ],
           [0.17 , 0.09 , 0.365, 0.38 ],
           [0.6  , 0.05 , 0.15 , 0.115],
           [0.085, 0.42 , 0.31 , 0.295],
           [0.04 , 0.17 , 0.345, 0.36 ],
           [0.305, 0.53 , 0.315, 0.335],
           [0.   , 0.   , 0.   , 0.   ],
           [0.   , 0.   , 0.   , 0.   ],
           [0.28 , 0.325, 0.39 , 0.38 ],
           [0.77 , 0.515, 0.23 , 0.175],
           [0.185, 0.15 , 0.325, 0.34 ],
           [0.6  , 0.175, 0.175, 0.13 ],
           [0.085, 0.555, 0.4  , 0.39 ],
           [0.47 , 0.655, 0.335, 0.255],
           [0.695, 0.725, 0.15 , 0.155],
           [0.285, 0.555, 0.35 , 0.265],
           [0.   , 0.   , 0.   , 0.   ],
           [0.32 , 0.49 , 0.23 , 0.22 ],
           [0.35 , 0.33 , 0.155, 0.165],
           [0.1  , 0.625, 0.36 , 0.35 ],
           [0.2  , 0.58 , 0.36 , 0.345],
           [0.   , 0.   , 0.   , 0.   ],
           [0.59 , 0.645, 0.265, 0.255],
           [0.36 , 0.61 , 0.255, 0.195],
           [0.695, 0.695, 0.25 , 0.26 ]], dtype=float32)




With the addition of the Pokemon classification, we need to change the loss function again. This time we use Categorical Crossentropy for the Pokemon type and Binary Crossentropy for the location and 'detect' values. A new weight coefficient is also added so we can tweak the algorithm to favour particular losses. Note also, that the Pokemon classification is only included in the loss when a Pokemon is present in the scene.

As in Lazy Programmer's lecture, we also want to weight the relative importance of detecting a Pokemon vs locating it in the image. The custom loss function can therefore be given as:

$$ loss = \alpha L_{bce}(Y_{[1..4]}, \hat{Y}_{[1..4]})Y_{[8]} + \beta L_{cce} (Y_{[5..7]}, \hat{Y}_{[5..7]})Y_{[8]} + \gamma L_{bce}(Y_{[8]}, \hat{Y}_{[8]})$$

Where:

&emsp;$$\alpha$$, $$\beta$$ and $$\gamma$$ are the weightings for the location, classification and detect losses  
&emsp;$$L_{bce}$$ represets the binary crossentropy loss function  
&emsp;$$L_{cce}$$ represets the categorical crossentropy loss function  
&emsp;$$Y_{[1..4]}$$ and $$\hat{Y}_{[1..4]}$$ represent the actual and predicted location outputs  
&emsp;$$Y_{[5..7]}$$ and $$\hat{Y}_{[5..7]}$$ represent the actual and predicted pokemon classification outputs  
&emsp;$$Y_{[8]}$$ and $$\hat{Y}_{[8]}$$ represent the actual and predicted detect outputs  

As previously, the 'detect' value is used to ignore values for loss in the location and classification outputs if no Pokemon is generated in the scene by multiplying it with those outputs. 

The custom loss is created as a class derived from the Keras Loss class as shown in the Tensorflow documentation at [https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss)


```python
class PokemonBinaryCrossEntropyError(tf.keras.losses.Loss):
    
    def __init__(self):
        super().__init__()
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy()
        self.weight_alpha = 1.
        self.weight_beta = 1.
        self.weight_gamma = 0.5
        
    def call(self, y_true, y_pred):
        position_loss = self.binary_crossentropy(y_true[:,0:4], y_pred[:, 0:4])
        pokemon_loss = self.categorical_crossentropy(y_true[:,4:7], y_pred[:, 4:7])
        detect_loss = self.binary_crossentropy(y_true[:, -1], y_pred[:, -1])

        rv = self.weight_alpha * position_loss * y_true[:, -1] + \
            self.weight_beta * pokemon_loss * y_true[:, -1] + \
            self.weight_gamma * detect_loss
        
        return rv
```

As before, we're using the Adam optimizer.


```python
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
```

Next we create the model as before.


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

x1 = tf.keras.layers.Dense(4, activation="sigmoid")(x)
x2 = tf.keras.layers.Dense(3, activation="softmax")(x)
x3 = tf.keras.layers.Dense(1, activation="sigmoid")(x)

x = tf.keras.layers.Concatenate()([x1, x2, x3])


# and build and compile it
model = tf.keras.Model(vgg16.input, x, name="Localization_Model")
model.compile(loss=PokemonBinaryCrossEntropyError(), optimizer=opt)
model.summary()
```

    Model: "Localization_Model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_5 (InputLayer)           [(None, 200, 200, 3  0           []                               
                                    )]                                                                
                                                                                                      
     block1_conv1 (Conv2D)          (None, 200, 200, 64  1792        ['input_5[0][0]']                
                                    )                                                                 
                                                                                                      
     block1_conv2 (Conv2D)          (None, 200, 200, 64  36928       ['block1_conv1[0][0]']           
                                    )                                                                 
                                                                                                      
     block1_pool (MaxPooling2D)     (None, 100, 100, 64  0           ['block1_conv2[0][0]']           
                                    )                                                                 
                                                                                                      
     block2_conv1 (Conv2D)          (None, 100, 100, 12  73856       ['block1_pool[0][0]']            
                                    8)                                                                
                                                                                                      
     block2_conv2 (Conv2D)          (None, 100, 100, 12  147584      ['block2_conv1[0][0]']           
                                    8)                                                                
                                                                                                      
     block2_pool (MaxPooling2D)     (None, 50, 50, 128)  0           ['block2_conv2[0][0]']           
                                                                                                      
     block3_conv1 (Conv2D)          (None, 50, 50, 256)  295168      ['block2_pool[0][0]']            
                                                                                                      
     block3_conv2 (Conv2D)          (None, 50, 50, 256)  590080      ['block3_conv1[0][0]']           
                                                                                                      
     block3_conv3 (Conv2D)          (None, 50, 50, 256)  590080      ['block3_conv2[0][0]']           
                                                                                                      
     block3_pool (MaxPooling2D)     (None, 25, 25, 256)  0           ['block3_conv3[0][0]']           
                                                                                                      
     block4_conv1 (Conv2D)          (None, 25, 25, 512)  1180160     ['block3_pool[0][0]']            
                                                                                                      
     block4_conv2 (Conv2D)          (None, 25, 25, 512)  2359808     ['block4_conv1[0][0]']           
                                                                                                      
     block4_conv3 (Conv2D)          (None, 25, 25, 512)  2359808     ['block4_conv2[0][0]']           
                                                                                                      
     block4_pool (MaxPooling2D)     (None, 12, 12, 512)  0           ['block4_conv3[0][0]']           
                                                                                                      
     block5_conv1 (Conv2D)          (None, 12, 12, 512)  2359808     ['block4_pool[0][0]']            
                                                                                                      
     block5_conv2 (Conv2D)          (None, 12, 12, 512)  2359808     ['block5_conv1[0][0]']           
                                                                                                      
     block5_conv3 (Conv2D)          (None, 12, 12, 512)  2359808     ['block5_conv2[0][0]']           
                                                                                                      
     block5_pool (MaxPooling2D)     (None, 6, 6, 512)    0           ['block5_conv3[0][0]']           
                                                                                                      
     flatten_4 (Flatten)            (None, 18432)        0           ['block5_pool[0][0]']            
                                                                                                      
     dense_12 (Dense)               (None, 4)            73732       ['flatten_4[0][0]']              
                                                                                                      
     dense_13 (Dense)               (None, 3)            55299       ['flatten_4[0][0]']              
                                                                                                      
     dense_14 (Dense)               (None, 1)            18433       ['flatten_4[0][0]']              
                                                                                                      
     concatenate_4 (Concatenate)    (None, 8)            0           ['dense_12[0][0]',               
                                                                      'dense_13[0][0]',               
                                                                      'dense_14[0][0]']               
                                                                                                      
    ==================================================================================================
    Total params: 14,862,152
    Trainable params: 147,464
    Non-trainable params: 14,714,688
    __________________________________________________________________________________________________


Then we train the model using the image generator.


```python
history = model.fit(seq, epochs=25, steps_per_epoch=50)
```

    Epoch 1/25
    50/50 [==============================] - 218s 4s/step - loss: 1.2190
    Epoch 2/25
    50/50 [==============================] - 231s 5s/step - loss: 1.0429
    Epoch 3/25
    50/50 [==============================] - 242s 5s/step - loss: 0.9266
    ...
    Epoch 23/25
    50/50 [==============================] - 250s 5s/step - loss: 0.4967
    Epoch 24/25
    50/50 [==============================] - 251s 5s/step - loss: 0.5064
    Epoch 25/25
    50/50 [==============================] - 249s 5s/step - loss: 0.4659


Plot the loss from the `model.fit()`.


```python
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(history.history['loss'])
plt.title('Loss')
plt.show()
```


    
![Training loss graph](/assets/localization-7-2.png)
    


Training can take some time so save out the model so it can be loaded from disk in future.


```python
model.save('saved-models/localization-7.h5', overwrite=True)
```

The `to_pil_rect()` function takes the y outputs from the model and converts the values into the correct coordinates for the image. We only want to do that for the location and size values though, not for the detect values. We also have to convert width and height into right and bottom values by adding them to the left and top values. I had to make a small modification to the code from the previous lecture to take into account the change in array shape.


```python
def to_pil_rect(y):
    rv = np.copy(y)
    rv[:, :4] = rv[:, :4] * IMAGE_DIM
    rv[:, 2:4] = rv[:, 2:4] + rv[:, 0:2]
    return rv
```

The sequence object can be used to create a set of images we can use for prediction.


```python
x, y = seq.__getitem__(0)
```

Use the trained model to make some predictions from the new batch.


```python
y_predicted = model.predict(x)
```

    1/1 [==============================] - 5s 5s/step



```python
print(f'{y_predicted}')
```

    [[5.04656017e-01 4.55304623e-01 1.74794450e-01 1.78234190e-01
      1.06209539e-01 8.83485675e-01 1.03048189e-02 9.93664682e-01]
     [1.88276991e-01 3.24535072e-01 2.56809354e-01 2.74745524e-01
      7.59936636e-03 9.91005301e-01 1.39535533e-03 9.98976707e-01]
     [2.13018909e-01 2.24052340e-01 4.52457279e-01 4.31486011e-01
      1.88289705e-04 9.99429047e-01 3.82720231e-04 9.99954641e-01]
     [2.46565834e-01 7.16228336e-02 1.31003067e-01 1.31195873e-01
      3.30779344e-01 1.42577708e-01 5.26642978e-01 8.56770694e-01]
     [1.59132048e-01 4.01893437e-01 4.89614993e-01 4.63358134e-01
      3.31727564e-02 1.77443586e-03 9.65052843e-01 9.99997318e-01]
     [6.78105474e-01 2.03255162e-01 2.19602332e-01 1.78929970e-01
      7.92663395e-01 1.67997375e-01 3.93391736e-02 9.89245176e-01]
     [1.81751430e-01 2.40194067e-01 3.45277101e-01 3.16684693e-01
      2.64868677e-01 1.25545962e-02 7.22576678e-01 9.99791384e-01]
     [2.40646586e-01 2.80282468e-01 1.56318620e-01 1.41708091e-01
      7.76130408e-02 1.96323860e-02 9.02754605e-01 9.84547496e-01]
     [3.47465783e-01 6.98431969e-01 2.87525296e-01 3.34806293e-01
      1.32248411e-03 9.93578196e-01 5.09919459e-03 9.99598980e-01]
     [3.10526580e-01 4.47279483e-01 2.50920177e-01 2.00122833e-01
      8.76009107e-01 6.60034195e-02 5.79873919e-02 9.49079454e-01]
     [5.83196998e-01 3.25574607e-01 4.64264989e-01 4.82457399e-01
      6.86105490e-02 2.12677149e-03 9.29262638e-01 9.99899566e-01]
     [2.21170664e-01 5.06985001e-02 1.90791577e-01 2.09309787e-01
      1.64904580e-01 9.90977138e-02 7.35997677e-01 9.72164452e-01]
     [7.40068927e-02 7.73600340e-02 3.10418960e-02 2.74883714e-02
      6.41008675e-01 2.14445353e-01 1.44545972e-01 6.21426702e-02]
     [3.24084997e-01 4.60575193e-01 3.34737629e-01 3.32199603e-01
      9.69564617e-01 3.55124264e-03 2.68841311e-02 9.98488963e-01]
     [9.93656814e-02 1.19284719e-01 5.23189865e-02 4.18379568e-02
      5.79423368e-01 3.09639961e-01 1.10936709e-01 1.70179933e-01]
     [1.54657170e-01 3.63637269e-01 3.72662336e-01 3.96043897e-01
      9.83639111e-05 9.99654412e-01 2.47313495e-04 9.99987841e-01]
     [3.33270401e-01 2.42063701e-01 1.46082684e-01 1.67098105e-01
      8.16336833e-03 9.85205829e-01 6.63073873e-03 9.79875922e-01]
     [1.83349580e-01 4.03087229e-01 3.28806192e-01 3.02652270e-01
      1.95785370e-02 9.70563889e-01 9.85753909e-03 9.96851265e-01]
     [2.86280334e-01 5.44624686e-01 2.01072574e-01 2.40558758e-01
      6.07172959e-03 2.06279196e-02 9.73300338e-01 9.98627603e-01]
     [4.75683719e-01 5.33514798e-01 1.20281138e-01 1.11370377e-01
      6.82376444e-01 1.56777278e-01 1.60846278e-01 8.45220089e-01]
     [5.71791232e-01 2.47321486e-01 1.37782648e-01 1.26485601e-01
      6.07684433e-01 1.75486520e-01 2.16829121e-01 9.23083127e-01]
     [7.65421271e-01 2.55167156e-01 2.70515710e-01 3.12473804e-01
      6.77680746e-02 2.75182705e-02 9.04713631e-01 9.96179581e-01]
     [5.57998121e-01 7.95688748e-01 2.57263005e-01 2.61681288e-01
      7.73218274e-02 8.66080940e-01 5.65972328e-02 9.82850730e-01]
     [5.01925468e-01 1.10105656e-01 2.48404637e-01 2.21483126e-01
      8.21596444e-01 2.31842604e-02 1.55219316e-01 9.68034327e-01]
     [4.79457602e-02 6.48897111e-01 3.49101841e-01 3.65611404e-01
      3.66840884e-03 9.92276251e-01 4.05539945e-03 9.98400807e-01]
     [2.75141448e-01 2.82722503e-01 1.78994805e-01 1.64931968e-01
      8.93788874e-01 3.96187305e-02 6.65923133e-02 9.74435627e-01]
     [2.92136580e-01 3.56279403e-01 8.70598257e-02 8.62448588e-02
      4.10262704e-01 4.23834324e-01 1.65902987e-01 6.30573392e-01]
     [9.19815674e-02 1.02594674e-01 5.97119294e-02 3.97881307e-02
      5.98771870e-01 1.99157968e-01 2.02070147e-01 1.56509906e-01]
     [3.74431282e-01 6.57120228e-01 2.58091688e-01 2.36282721e-01
      9.53155816e-01 1.70275997e-02 2.98166275e-02 9.98570442e-01]
     [4.42867398e-01 6.56009242e-02 3.21117252e-01 3.57662439e-01
      4.44353418e-03 9.89962995e-01 5.59346145e-03 9.99543667e-01]
     [4.36794072e-01 5.31198144e-01 2.80984372e-01 2.55575448e-01
      1.88451737e-01 2.53768545e-02 7.86171436e-01 9.94324028e-01]
     [8.48135799e-02 2.80728161e-01 2.64773667e-01 2.66234696e-01
      2.43458245e-02 3.91971227e-03 9.71734524e-01 9.99072850e-01]]


Create the boxes for both the actual boxes and the predicted ones.


```python
predicted_boxes = to_pil_rect(y_predicted)
actual_boxes = to_pil_rect(y)
```

Next we create a function that takes each of the images used in the predictions and adds the predicted bounding boxes. I did this using the Pillow library but it's probably pretty straightforward to use Numoy or tf functions to do it too.


```python
def add_bounding_boxes():
    """
    Adds the bounding boxes to the images used in the prediction.
    """
    rv = np.zeros((seq.batch_size, IMAGE_DIM, IMAGE_DIM, 3), dtype=np.uint8)
    for i in range(seq.batch_size):
        im = Image.fromarray(np.uint8(x[i] * 255), mode='RGB')

        draw = ImageDraw.Draw(im)
        
        # add the bounding rectangle if something was detected
        if predicted_boxes[i, -1] > 0.5:

            draw.rectangle(
                [predicted_boxes[i, 1], 
                 predicted_boxes[i, 0], 
                 predicted_boxes[i, 3], 
                 predicted_boxes[i, 2]], outline='green')
        
        rv[i] = np.array(im).astype(dtype=np.uint8)

    return rv
```


```python
final_images = add_bounding_boxes()
```

Now we can draw some of the images that were used to create the predictions and overlay the localization rectangles on top of them as before.


```python
pokemon_categories = ["Charminder", "Bulbasaur", "Squirtle"]
def get_pokemon_name_from_prediction(y):
    a = y[4:7]
    d = y[-1] > 0.5
    return pokemon_categories[np.argmax(a)] if d else "None"
```


```python
plt.rcParams["figure.figsize"] = (10,10)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    im_idx = random.randint(0, seq.batch_size - 1)
    ax.imshow(final_images[im_idx])
    ax.set_axis_off()
    ax.set_title(f'{get_pokemon_name_from_prediction(predicted_boxes[im_idx])}')
```


    
![Output from predictions showing images and bounding boxes](/assets/localization-7-3.png)
    


In the output above you can see the green boxes that are the results of the predictions. You can see that some of the predictions are a bit off, so again so it might be worth training for more epochs to see if a bit more accuracy can be obtained that way. Note that the loss curve from training looked as though it could be pushed a bit further.

This is the end of the Localization lectures from the course. Overall, it's been an interesting exercise to work through them all one-by-one and gradually build up to the final code. It's taught me  a lot about using the Tensorflow library, how to build custom models and make use of transfer learning. It also showed me how to build models with layers that use different activation functions using the `Concatenate` class and how to write custom loss functions by deriving from the `Loss` class in Tensorflow.

For my next project, I'm going to see if I can take a paper from [https://arxiv.org/](https://arxiv.org/) and I'll post progress on that as I go.
