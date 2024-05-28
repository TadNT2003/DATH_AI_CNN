import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# print("Check TensorFlow:", tf.__version__)
# print("Check Keras:", kr.__version__)
# dataset_urls = "file:\\D:\\Pygame projects\\DATH_TTNT\\lfw-funneled.tgz"
# archive = tf.keras.utils.get_file('lfw_funneled.tgz', origin=dataset_urls, extract=True)
# data_dir = pathlib.Path(archive).with_suffix('')

# dataset directory
data_dir = "D:\\Pygame projects\\DATH_TTNT\\Bird_recog"
# data_dir = pathlib.Path(data_dir).with_suffix('')
img_height = 224
img_width = 224
batch_size = 64

# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)
# print(data_dir)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "\\train",
    seed=42,
    color_mode='rgb',
    # label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "\\test",
    seed=42,
    color_mode='rgb',
    # label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)
predict_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "\\valid",
    seed=42,
    color_mode='rgb',
    # label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

# Visualize data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Configure dataset for performance
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Since the pixel is already in range[0,1] so no need to normalize

# To find the shape of input
# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break

# Create Keras model using Sequential
num_classes = len(class_names)
model = Sequential([
    # Data preprocessing
    layers.Resizing(112, 112, input_shape= (224, 224, 3) ),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    # Yoinking from tutorial, not optimized for accuracy
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.05),
    layers.Dense(num_classes)
        
])

# Model summary
# model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model using train_ds
history = model.fit(
    train_ds,
    validation_data = test_ds,
    epochs = 10
)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()
plt.show()

result = model.evaluate(predict_ds, batch_size=batch_size)
print("test loss, test acc", result)

img_url = ".\\Bird_recog\\valid\\ABBOTTS BABBLER\\1.jpg"
img_path = tf.keras.utils.get_file('ABBOTTS BABBLER', origin=img_url)

img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# pred = model.predict(predict_ds)
# score = tf.nn.softmax(pred[0])
# plt.imshow()

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )