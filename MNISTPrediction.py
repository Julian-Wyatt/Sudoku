# import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np
import tensorflowjs as tfjs

tf.enable_v2_behavior()
# (ds_train, ds_test), ds_info = tfds.load(
#     'mnist',
#     split=['train', 'test'],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(ds_train.take(9)):
#     image_ = np.array(image)
#     print(image_.squeeze().shape)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image_.squeeze(),cmap='gray')
#     plt.title(int(label))
#     plt.axis("off")
# # plt.show()
#
# def normalize_img(image, label):
#   """Normalizes images: `uint8` -> `float32`."""
#   return tf.cast(image, tf.float32) / 255., label
#
# ds_train = ds_train.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.batch(128)
# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
#
#
#
# ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

img_height = 28
img_width = 28
batch_size = 32
data_dir = "./fontImages/"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
    color_mode='grayscale',
  validation_split=0.35,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
color_mode='grayscale',
  validation_split=0.35,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (5, 5), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.GaussianNoise(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    verbose=1
)

# model.save("./Model/saved28x28NumberPredictor")
tfjs.converters.save_keras_model(model, "./Model/js")