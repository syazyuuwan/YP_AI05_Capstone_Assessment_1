#%%
#1. Import Libraries

import shutil
import numpy as np
import os, datetime
import tensorflow as tf
import keras.api._v2.keras as keras
from matplotlib import pyplot as plt
from keras import layers, optimizers, losses, callbacks, applications
# %%
#2. Data Loading

#Load data from local directory
DATA_PATH = os.path.join(os.getcwd(),'Data')
data = keras.utils.image_dataset_from_directory(DATA_PATH,
                                                shuffle=False)
class_names = data.class_names
# %%
#Extract a sample of images for deployment to avoid polluting the dataset

images_per_class = 5
#Create a directory to store the exported images with class names.
SAMPLE_PATH = os.path.join(os.getcwd(),'Sample')
os.makedirs(SAMPLE_PATH, exist_ok=True)

#Create a directory to temporarily store images to be removed from the original dataset.
REMOVE_PATH = os.path.join(os.getcwd(),'Temp')
os.makedirs(REMOVE_PATH, exist_ok=True)

#Iterate through the dataset, export the specified number of images per class, and remove them from the original dataset.
for images, labels in data:
    i = 0
    for image, label in zip(images, labels):
        class_name = class_names[label]
        class_directory = os.path.join(SAMPLE_PATH, class_name)
        os.makedirs(class_directory, exist_ok=True)

        #Count the exported images for the current class.
        class_images_count = len(os.listdir(class_directory))

        #Check if we have exported the desired number of images for this class.
        if class_images_count < images_per_class:
            # Generate a unique filename based on image count.
            filename = f"0000{i+1}.jpg"
            i+=1
            #Save the image to the class directory.
            image_path = os.path.join(class_directory, class_name+filename)
            keras.preprocessing.image.save_img(image_path, image.numpy())

            #Remove the image from the original dataset.
            removed_image_path = os.path.join(REMOVE_PATH, filename)
            shutil.move(os.path.join(DATA_PATH, class_name, filename), removed_image_path)

            #Check if we have exported the desired number of images for all classes.
            if all(len(os.listdir(os.path.join(SAMPLE_PATH, class_name))) >= images_per_class for class_name in class_names):
                break

#Print a message indicating that the desired number of images per class has been exported.
print(f"{images_per_class} images per class exported to {SAMPLE_PATH}")

#Remove the temporary directory containing removed images.
shutil.rmtree(REMOVE_PATH)
# %%
#Re-load the dataset after extracting sample

data = keras.utils.image_dataset_from_directory(DATA_PATH,
                                                batch_size=64,
                                                shuffle=True,
                                                seed=42)
#%%
#3. Inspect data 
plt.figure(figsize=(10, 10))
for images, labels in data.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#%%
#4. Split data to train, val, test

train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.1)

train_data = data.take(train_size)
val_data =  data.skip(train_size).take(val_size)
test_data = data.skip(train_size+val_size).take(test_size)
# %%
#5. Convert to prefetch datasets to speed up model training

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)
# %%
#6. Create a sequential 'model' for data augmentation

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2)
])
# %%
#7. Apply data augmentation on an image

for image, _ in train_data.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 227)
    plt.axis('off')
# %%
#8. Define a layer for data normalization/rescaling

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# %%
#9. Instantiate base model for transfer learning

IMG_SHAPE =(256,256,3)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,
                                      include_top=False,
                                      weights='imagenet')
# %%
#Freeze the entire feature extractor
base_model.trainable = False
# %%
#Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#Create the output layer
output_layer = layers.Dense(len(class_names),
                            activation='softmax')
#Build the entire pipeline using functional API

#input
inputs = keras.Input(shape=IMG_SHAPE)

#Data augmentation model
x = data_augmentation(inputs)

#Data rescaling layer
x = preprocess_input(x)

#Transfer learning feature extractor
x = base_model(x, training = False)

#Final extracted features
x = global_avg(x)

#Classification layer
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

#Build the model

model = keras.Model(inputs = inputs, outputs = outputs)
# %%
#10. Compile model

optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# %%
#Create Tensorboard callback object
base_log_path=r"tensorboard_logs\assignment1"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)

early_stopping = callbacks.EarlyStopping(patience=2)
# %%
#11. Model Training

EPOCHS = 10
history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=EPOCHS,
                    callbacks=[tb, early_stopping])
# %%
#12. Model Evaluation

loss1, acc1 = model.evaluate(test_data)
print("Evaluation after training")
print("Loss =",loss1)
print("Accuracy =",acc1)
# %%
#13. Model Saving

model.save(os.path.join(os.getcwd(),'models', 'ASSESSMENT_1_MODEL.h5'))
#%%
#View model architecture
keras.utils.plot_model(model)
