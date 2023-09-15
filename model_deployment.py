#%%
#1. Import packages
import os
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from matplotlib import pyplot as plt
#%%
#2. Model Loading
SAMPLE_PATH = os.path.join(os.getcwd(),'Sample')
sample_data = keras.utils.image_dataset_from_directory(SAMPLE_PATH)
loaded_model = tf.keras.models.load_model(os.path.join(os.getcwd(),'models', 'ASSESSMENT_1_MODEL.h5'))
#%%
#3. Model Deployment

#Predicting using the model
image_batch, label_batch = sample_data.as_numpy_iterator().next()
predictions = loaded_model.predict_on_batch(image_batch)
# %%
#Mapping sample data with actual and precited labels for the the following plots
class_names = sample_data.class_names
prediction_index = np.argmax(predictions, axis=1)
label_map = {i:names for i,names in enumerate(class_names)}
prediction_label = [label_map[i] for i in prediction_index]
label_class_list = [label_map[i] for i in label_batch]    
# %%
#Plotting results
plt.figure(figsize=(15,15))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(f"Actual:{label_class_list[i]} | Prediction:{prediction_label[i]}")
    plt.axis('off')
    plt.grid('off')

# %%
