---
title: "Image Classifier"
date: 2021-04-28
tags: [image classification, data science, CNN, neural networks, python]
header:
  image: '/images/cnn.jpeg'
excerpt: "Creating an image classifier through a convolutional neural network"
mathjax: "true"
---

## What is image classification?

### Image classification is using a computer to analyze an image and identify which class appears in the image. A class is the category or label of the object being identified. In this image classifier, the class is one of the following: sea, mountain, forest, glacier, building or street. When given an image, the classifier will assign probabilities of each class appearing in the image. The class with the highest probability will be assigned to the image.

## How does image classification work?

### I used a convolutional neural network (CNN) to create the classifier. A CNN works by using pixel data to extract features from the image. The image is convolved into a 2D matrix using filters that slide over a specified range of pixels at a time. A rectified linear activation function (ReLU) is applied on the input and hidden layers of the model to increase non-linearity of the features. Pooling, dropout, and normalization are all used on the hidden layers. Pooling reduces the dimensions, dropout randomly removes neurons to prevent the model from overfitting, and normalization is used to balance the scale of each feature. The features are flattened into a 1D vector and input through a fully connected (Dense) layer. A softmax function is applied on the output layer, which converts the previous layer outputs into probability distributions for each class. This will tell us which class the image most likely contains.



# Overview of Findings
<iframe src="https://bellevueuniversity-my.sharepoint.com/:p:/g/personal/sherbstreit_my365_bellevue_edu/ETZgR5lBsZlHknNx0Y_Pbv4BpIKiCHsHhuNjLKzMMvK_rA?e=82tMuv&amp;action=embedview&amp;wdAr=1.7777777777777777" width="850px" height="421px" frameborder="0">This is an embedded <a target="_blank" href="https://office.com">Microsoft Office</a> presentation, powered by <a target="_blank" href="https://office.com/webapps">Office</a>.</iframe>

# The code used

```python
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import tensorflow as tf                
from tqdm import tqdm
import cv2
from tensorflow.keras import regularizers
```

```python
def get_images(img_path):
    label = _
    Images = []
    Labels = []
    
    for labels in os.listdir(img_path):
        #ordinal encoding of the categorical labels
        if labels == 'buildings': 
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'glacier':
            label = 2
        elif labels == 'mountain':
            label = 3
        elif labels == 'sea':
            label = 4
        elif labels == 'street':
            label = 5
        
        for image_file in os.listdir(img_path+labels):
            image = cv2.imread(img_path+labels+r'/'+image_file)
            #resize all images to 150x150
            image = cv2.resize(image, (150,150))
            #add to image list
            Images.append(image)
            #add ordinally encoded label
            Labels.append(label)
    #randomize images
    return shuffle(Images, Labels, random_state=72)
```


```python
train_images, train_labels = get_images('seg_train/')
```


```python
test_images, test_labels = get_images('seg_test/')
```


```python
# visualize some examples of the training data
classes = ['buildings','forest','glacier','mountain','sea','street']   

fig = plt.figure(figsize=(20,10))
fig.suptitle('Training Images', fontsize=26)
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    # Started at 20 to show more mountain and glacier images
    plt.imshow(train_images[20+i], cmap=plt.cm.binary)
    plt.xlabel(classes[train_labels[20+i]])
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_4_0.jpg" alt="linearly separable data">



## We can see that some of these images appear to have contradicting or incorrect labels. This is most notable in the mountain and glacier classes. Many photos also include more than one class.



```python
#visualize some test images
fig = plt.figure(figsize=(20,10))
fig.suptitle('Test Images', fontsize=26)
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[20+i], cmap=plt.cm.binary)
    plt.xlabel(classes[test_labels[20+i]])
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_6_0.jpg" alt="linearly separable data">




```python
x_train = np.array(train_images)
y_train = np.array(train_labels)
```


```python
x_test = np.array(test_images)
y_test = np.array(test_labels)
```


```python
model = Sequential()
model.add(Conv2D(16,(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((5,5)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
model.add(MaxPooling2D((5,5)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_3 (Conv2D)            (None, 148, 148, 16)      448       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 146, 146, 32)      4640      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 29, 29, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 27, 27, 32)        9248      
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 25, 25, 64)        18496     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 5, 5, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1600)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                102464    
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 64)                256       
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 135,942
    Trainable params: 135,814
    Non-trainable params: 128
    _________________________________________________________________



```python
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
```


```python
train_model = model.fit(x_train, y_train, epochs=30,
                           validation_data=(x_test, y_test))
```

    Epoch 1/30
    439/439 [==============================] - 169s 385ms/step - loss: 1.6856 - accuracy: 0.4508 - val_loss: 0.9401 - val_accuracy: 0.7013
    Epoch 2/30
    439/439 [==============================] - 174s 397ms/step - loss: 0.9104 - accuracy: 0.7011 - val_loss: 0.7738 - val_accuracy: 0.7423
    Epoch 3/30
    439/439 [==============================] - 171s 390ms/step - loss: 0.7038 - accuracy: 0.7745 - val_loss: 0.8630 - val_accuracy: 0.7123
    Epoch 4/30
    439/439 [==============================] - 172s 391ms/step - loss: 0.6199 - accuracy: 0.7996 - val_loss: 0.5131 - val_accuracy: 0.8423
    Epoch 5/30
    439/439 [==============================] - 170s 387ms/step - loss: 0.5655 - accuracy: 0.8185 - val_loss: 0.5731 - val_accuracy: 0.8097
    Epoch 6/30
    439/439 [==============================] - 174s 397ms/step - loss: 0.6105 - accuracy: 0.7974 - val_loss: 0.5977 - val_accuracy: 0.7957
    Epoch 7/30
    439/439 [==============================] - 173s 395ms/step - loss: 0.5214 - accuracy: 0.8313 - val_loss: 0.5032 - val_accuracy: 0.8377
    Epoch 8/30
    439/439 [==============================] - 164s 374ms/step - loss: 0.5061 - accuracy: 0.8337 - val_loss: 0.4792 - val_accuracy: 0.8433
    Epoch 9/30
    439/439 [==============================] - 161s 366ms/step - loss: 0.4825 - accuracy: 0.8392 - val_loss: 0.4598 - val_accuracy: 0.8593
    Epoch 10/30
    439/439 [==============================] - 159s 363ms/step - loss: 0.4583 - accuracy: 0.8488 - val_loss: 0.6223 - val_accuracy: 0.8023
    Epoch 11/30
    439/439 [==============================] - 160s 364ms/step - loss: 0.4701 - accuracy: 0.8502 - val_loss: 0.4581 - val_accuracy: 0.8570
    Epoch 12/30
    439/439 [==============================] - 160s 363ms/step - loss: 0.4563 - accuracy: 0.8511 - val_loss: 0.6920 - val_accuracy: 0.7767
    Epoch 13/30
    439/439 [==============================] - 160s 363ms/step - loss: 0.4287 - accuracy: 0.8655 - val_loss: 0.7445 - val_accuracy: 0.7337
    Epoch 14/30
    439/439 [==============================] - 159s 363ms/step - loss: 0.4206 - accuracy: 0.8636 - val_loss: 0.5327 - val_accuracy: 0.8197
    Epoch 15/30
    439/439 [==============================] - 160s 363ms/step - loss: 0.4125 - accuracy: 0.8656 - val_loss: 0.4468 - val_accuracy: 0.8593
    Epoch 16/30
    439/439 [==============================] - 158s 360ms/step - loss: 0.3948 - accuracy: 0.8772 - val_loss: 0.5408 - val_accuracy: 0.8247
    Epoch 17/30
    439/439 [==============================] - 158s 361ms/step - loss: 0.4047 - accuracy: 0.8718 - val_loss: 0.5565 - val_accuracy: 0.8307
    Epoch 18/30
    439/439 [==============================] - 159s 361ms/step - loss: 0.4292 - accuracy: 0.8615 - val_loss: 0.4281 - val_accuracy: 0.8730
    Epoch 19/30
    439/439 [==============================] - 159s 362ms/step - loss: 0.4071 - accuracy: 0.8669 - val_loss: 0.4430 - val_accuracy: 0.8650
    Epoch 20/30
    439/439 [==============================] - 158s 360ms/step - loss: 0.3807 - accuracy: 0.8769 - val_loss: 0.5148 - val_accuracy: 0.8503
    Epoch 21/30
    439/439 [==============================] - 159s 362ms/step - loss: 0.3793 - accuracy: 0.8823 - val_loss: 0.4586 - val_accuracy: 0.8513
    Epoch 22/30
    439/439 [==============================] - 161s 367ms/step - loss: 0.4019 - accuracy: 0.8681 - val_loss: 0.4763 - val_accuracy: 0.8543
    Epoch 23/30
    439/439 [==============================] - 164s 373ms/step - loss: 0.3933 - accuracy: 0.8786 - val_loss: 0.4401 - val_accuracy: 0.8637
    Epoch 24/30
    439/439 [==============================] - 165s 375ms/step - loss: 0.3524 - accuracy: 0.8918 - val_loss: 0.4288 - val_accuracy: 0.8780
    Epoch 25/30
    439/439 [==============================] - 165s 377ms/step - loss: 0.3513 - accuracy: 0.8886 - val_loss: 0.4782 - val_accuracy: 0.8447
    Epoch 26/30
    439/439 [==============================] - 166s 378ms/step - loss: 0.3711 - accuracy: 0.8803 - val_loss: 0.4467 - val_accuracy: 0.8653
    Epoch 27/30
    439/439 [==============================] - 167s 379ms/step - loss: 0.3564 - accuracy: 0.8892 - val_loss: 0.4191 - val_accuracy: 0.8750
    Epoch 28/30
    439/439 [==============================] - 166s 377ms/step - loss: 0.3678 - accuracy: 0.8858 - val_loss: 0.4857 - val_accuracy: 0.8513
    Epoch 29/30
    439/439 [==============================] - 166s 378ms/step - loss: 0.3363 - accuracy: 0.8962 - val_loss: 0.4509 - val_accuracy: 0.8593
    Epoch 30/30
    439/439 [==============================] - 164s 373ms/step - loss: 0.3262 - accuracy: 0.8988 - val_loss: 0.4411 - val_accuracy: 0.8693



```python
# visualize loss and accuracy
acc = train_model.history['accuracy']
val_acc = train_model.history['val_accuracy']
loss = train_model.history['loss']
val_loss = train_model.history['val_loss']

epochs_range = range(30)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_13_0.jpg" alt="linearly separable data">

    



```python
predictions = model.predict(x_test)
pred_labels = np.argmax(predictions, axis = 1)
```


```python
#visualize predictions
import seaborn as sns
CM = confusion_matrix(y_test, pred_labels)
ax = plt.axes()
sns.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=classes, 
           yticklabels=classes, ax = ax)
ax.set_title('Confusion matrix')
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_15_0.jpg" alt="linearly separable data">

    


## We can see that the model did poor a job differentiating between the glacier and mountain classes. 


```python
#load some images from a google search for each category
from tensorflow.keras.preprocessing import image
```


```python
forest_img = image.load_img('Forest.jpeg', target_size=(150, 150))
input_arr = image.img_to_array(forest_img)
input_arr = np.array([input_arr])
forest_pred = model.predict(input_arr)
```


```python
sea_img = image.load_img('Sea.jpeg', target_size=(150, 150))
input_arr = image.img_to_array(sea_img)
input_arr = np.array([input_arr])
sea_pred = model.predict(input_arr)
```


```python
streets_img = image.load_img('Streets.jpeg', target_size=(150, 150))
input_arr = image.img_to_array(streets_img)
input_arr = np.array([input_arr])
street_pred = model.predict(input_arr)
```


```python
glacier_img = image.load_img('Glacier.jpeg', target_size=(150, 150))
input_arr = image.img_to_array(glacier_img)
input_arr = np.array([input_arr])
glacier_pred = model.predict(input_arr)
```


```python
mount_img = image.load_img('Mountains.jpeg', target_size=(150, 150))
input_arr = image.img_to_array(mount_img)
input_arr = np.array([input_arr])
mountain_pred = model.predict(input_arr)
```


```python
build_img = image.load_img('Buildings.jpeg', target_size=(150, 150))
input_arr = image.img_to_array(build_img)
input_arr = np.array([input_arr])
buildings_pred = model.predict(input_arr)
```


```python
trick_img = image.load_img('Sea-Mountains.jpeg', target_size=(150, 150))
input_arr = image.img_to_array(trick_img)
input_arr = np.array([input_arr])
sea_mount_pred = model.predict(input_arr)
```


```python
#visualize the images from the Google search
classes = ['Forest','Sea','Streets','Glacier','Mountains','Buildings','Sea-Mountains']
images = [forest_img, sea_img, streets_img, glacier_img, mount_img, build_img, trick_img]

fig = plt.figure(figsize=(12,6))
fig.suptitle('Google Test Images', fontsize=16)
for i in range(7):
    plt.subplot(2,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(classes[i])
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_25_0.jpg" alt="linearly separable data">




```python
predictions = [forest_pred, sea_pred, street_pred, glacier_pred, 
               mountain_pred, buildings_pred, sea_mount_pred]
```


```python
# Check how well the model predicts new data
df = pd.DataFrame(np.concatenate(predictions),columns=classes)
df['Image Type'] = ['Forest','Sea','Streets','Glacier','Mountains',
                   'Buildings','Sea-Mountains']
```


```python
df=df.set_index('Image Type')
df.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>buildings</th>
      <th>forest</th>
      <th>glacier</th>
      <th>mountain</th>
      <th>sea</th>
      <th>street</th>
    </tr>
    <tr>
      <th>Image Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Forest</th>
      <td>0.005</td>
      <td>0.988</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>Sea</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.010</td>
      <td>0.034</td>
      <td>0.955</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>Streets</th>
      <td>0.004</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.996</td>
    </tr>
    <tr>
      <th>Glacier</th>
      <td>0.007</td>
      <td>0.936</td>
      <td>0.013</td>
      <td>0.003</td>
      <td>0.005</td>
      <td>0.037</td>
    </tr>
    <tr>
      <th>Mountains</th>
      <td>0.299</td>
      <td>0.000</td>
      <td>0.018</td>
      <td>0.463</td>
      <td>0.204</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>Buildings</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>Sea-Mountains</th>
      <td>0.010</td>
      <td>0.664</td>
      <td>0.160</td>
      <td>0.074</td>
      <td>0.091</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>



### The model did very well in the forest, sea, streets and building classes. The model still has a hard time reading new glacier or mountain images. When presented with a "trick" image containing sea, mountains and forest, the model identifies this as forest. While this is not totally incorrect, the majority of the image is sea and I would have preferred it to be labeled as sea. 
<br />
# Future Improvements

### Revisiting the label assignments in the training and validation data could improve the precision and specificity of the model. I believe some of the mountain and glacier images were labeled incorrectly, leading to poor fitting on these classes. Adding secondary label tags to identify other categories inclued in the image would also be beneficial. 


```python

```
