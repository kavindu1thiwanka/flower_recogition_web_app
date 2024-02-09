import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models

# Set Hyperparameters and Paths
batch_size = 32
img_size = (299, 299)
epochs = 20
num_classes = 10  # number of flower categories

train_data_dir = 'flowers/train'
validation_data_dir = 'flowers/validation'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load InceptionV3 Base Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Build Custom Model on Top of InceptionV3
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Use the compatible optimizer
optimizer = tf.train.AdamOptimizer()  # You can adjust learning_rate if needed
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set Up Data Generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the Model
model.save('custom_flower_recognition_model.h5')
