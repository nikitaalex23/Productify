
from google.colab import drive
import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split

drive.mount('/content/drive')

zip_path = "/content/drive/My Drive/images_cleaned.zip"
extract_dir = "/content/images_cleaned"
image_dir = "/content/images_cleaned/images"

if not os.path.exists(image_dir):
    print("Extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✅ Extraction complete.")
else:
    print("✅ Images already extracted.")

image_files_in_drive = os.listdir(image_dir)
print(f"Total images found: {len(image_files_in_drive)}")
print("Example files:", image_files_in_drive[:10])

shape_file = "/content/shape_ann.txt"
fabric_file = "/content/fabric_ann_cleaned.txt"
pattern_file = "/content/pattern_ann_cleaned.txt"

# Loading Annotations
def load_annotation_file(file_path):
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            items = line.strip().split()
            try:
                data[items[0]] = list(map(int, items[1:]))
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
    return data

shape_data = load_annotation_file(shape_file)
fabric_data = load_annotation_file(fabric_file)
pattern_data = load_annotation_file(pattern_file)

print(f"Loaded annotations -> Shape: {len(shape_data)}, Fabric: {len(fabric_data)}, Pattern: {len(pattern_data)}")


image_files, shape_labels, fabric_labels, pattern_labels = [], [], [], []

for img_name in shape_data.keys():
    img_path = os.path.join(image_dir, img_name)
    if os.path.exists(img_path) and img_name in fabric_data and img_name in pattern_data:
        image_files.append(img_path)
        shape_labels.append(shape_data[img_name])
        fabric_labels.append(fabric_data[img_name])
        pattern_labels.append(pattern_data[img_name])

print(f"✅ Total valid images: {len(image_files)}")

train_paths, test_paths, y_train_shape, y_test_shape, y_train_fabric, y_test_fabric, y_train_pattern, y_test_pattern = train_test_split(
    image_files, shape_labels, fabric_labels, pattern_labels, test_size=0.2, random_state=42
)

print(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")

#Image Processing
IMG_SIZE = (224, 224)
num_classes_shape = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3]
num_classes_fabric = [8, 8, 8]
num_classes_pattern = [8, 8, 8]

def process_path(file_path, label_shape, label_fabric, label_pattern):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3, try_recover_truncated=True)
    image = tf.image.resize(image, IMG_SIZE) / 255.0

    shape_enc = [tf.one_hot(label_shape[i], num_classes_shape[i]) for i in range(len(num_classes_shape))]
    fabric_enc = [tf.one_hot(label_fabric[i], num_classes_fabric[i]) for i in range(len(num_classes_fabric))]
    pattern_enc = [tf.one_hot(label_pattern[i], num_classes_pattern[i]) for i in range(len(num_classes_pattern))]

    output = {}
    for idx, enc in enumerate(shape_enc): output[f'shape_{idx}'] = enc
    for idx, enc in enumerate(fabric_enc): output[f'fabric_{idx}'] = enc
    for idx, enc in enumerate(pattern_enc): output[f'pattern_{idx}'] = enc

    return image, output

#Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.2)
])

def get_dataset(paths, shapes, fabrics, patterns, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((paths, shapes, fabrics, patterns))
    dataset = dataset.shuffle(1000).map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = get_dataset(train_paths, y_train_shape, y_train_fabric, y_train_pattern)
test_dataset = get_dataset(test_paths, y_test_shape, y_test_fabric, y_test_pattern)

print("✅ Datasets ready!")

#Using ConvNeXtBase as Base Model
base_model = applications.ConvNeXtBase(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs, output_names = [], []

for idx, num_class in enumerate(num_classes_shape):
    out = layers.Dense(num_class, activation='softmax', name=f'shape_{idx}')(x)
    outputs.append(out)
    output_names.append(f'shape_{idx}')

for idx, num_class in enumerate(num_classes_fabric):
    out = layers.Dense(num_class, activation='softmax', name=f'fabric_{idx}')(x)
    outputs.append(out)
    output_names.append(f'fabric_{idx}')

for idx, num_class in enumerate(num_classes_pattern):
    out = layers.Dense(num_class, activation='softmax', name=f'pattern_{idx}')(x)
    outputs.append(out)
    output_names.append(f'pattern_{idx}')

model = models.Model(inputs=base_model.input, outputs=outputs)
model.summary()


loss_dict = {name: tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1) for name in output_names}
metrics_dict = {name: 'accuracy' for name in output_names}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=loss_dict, metrics=metrics_dict)

#Training Model (Initial Phase)
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
])

#Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=loss_dict, metrics=metrics_dict)

history_fine = model.fit(train_dataset, validation_data=test_dataset, epochs=5)
    
model.save('/content/drive/My Drive/convnext_fashion_model.keras')
print("✅ ConvNeXt model trained, fine-tuned, and saved!")