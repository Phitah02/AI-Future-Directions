#!/usr/bin/env python
# coding: utf-8

# # Edge AI Prototype: Garbage Classification
# 
# This notebook implements a lightweight image classification model for recognizing recyclable items using TensorFlow Lite.
# 
# ## Goals:
# 1. Train a MobileNetV2-based model on the garbage classification dataset.
# 2. Convert the model to TensorFlow Lite.
# 3. Test the model and explain Edge AI benefits.

# In[9]:


# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


# ## Step 1: Data Preparation

# In[10]:


# Define dataset path
dataset_path = 'Task 1/garbage_classification'

# Get class names
classes = os.listdir(dataset_path)
classes.sort()
print(f"Classes: {classes}")
print(f"Number of classes: {len(classes)}")

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# Data generator for validation (no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# For testing, we'll use a separate split
# Note: In a real scenario, you'd have a separate test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# ## Step 2: Model Training

# In[11]:


# Load MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()


# In[ ]:


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
epochs = 20
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[5]:


# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Classification report
print(classification_report(true_classes, predicted_classes, target_names=classes))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# ## Step 3: TensorFlow Lite Conversion and Testing

# In[6]:


# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply default optimizations
tflite_model = converter.convert()

# Save the TFLite model
with open('garbage_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")


# In[7]:


# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='garbage_classifier.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])


# In[8]:


# Test TFLite model on a few sample images
def test_tflite_model(interpreter, test_images, test_labels, num_samples=5):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    total_time = 0

    for i in range(min(num_samples, len(test_images))):
        # Prepare input
        input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()

        inference_time = end_time - start_time
        total_time += inference_time

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        true_class = test_labels[i]

        print(f"Sample {i+1}: Predicted: {classes[predicted_class]}, True: {classes[true_class]}, Time: {inference_time:.4f}s")

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / min(num_samples, len(test_images))
    avg_time = total_time / min(num_samples, len(test_images))
    print(f"\nTFLite Test Accuracy: {accuracy:.4f}")
    print(f"Average Inference Time: {avg_time:.4f}s")

    return accuracy, avg_time

# Get some test images and labels
test_images, test_labels = next(test_generator)
test_labels = np.argmax(test_labels, axis=1)

# Test the TFLite model
tflite_accuracy, avg_inference_time = test_tflite_model(interpreter, test_images, test_labels)


# ## Step 4: Report and Explanation
# 
# ### Accuracy Metrics:
# - Training Accuracy: {history.history['accuracy'][-1]:.4f}
# - Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}
# - Test Accuracy: {test_accuracy:.4f}
# - TFLite Test Accuracy: {tflite_accuracy:.4f}
# - Average Inference Time: {avg_inference_time:.4f}s
# 
# ### Deployment Steps:
# 1. Train the model using this notebook.
# 2. Convert to TFLite as shown above.
# 3. Deploy on edge devices (Raspberry Pi, mobile phones, etc.).
# 4. Use the TFLite interpreter for inference.
# 
# ### Edge AI Benefits for Real-Time Applications:
# Edge AI refers to running AI models directly on edge devices rather than in the cloud. Key benefits include:
# 
# 1. **Low Latency**: Processing happens locally, reducing response times for real-time applications.
# 2. **Privacy**: Data doesn't need to be sent to the cloud, keeping sensitive information local.
# 3. **Offline Operation**: Works without internet connectivity.
# 4. **Reduced Bandwidth**: Less data needs to be transmitted.
# 5. **Cost Efficiency**: Lower cloud computing costs.
# 
# In this garbage classification example, Edge AI enables real-time sorting in recycling facilities, privacy-preserving waste monitoring in smart cities, and offline operation in remote areas.
