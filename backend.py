import warnings
import tensorflow as tf
import torch
import os

# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs except errors
# Suppress TensorFlow deprecation warnings
tf.get_logger().setLevel('ERROR')

# Suppress PyTorch specific warnings related to 'pretrained' and 'FutureWarnings'
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# Suppress all warnings from TensorFlow and PyTorch to avoid any unnecessary logs
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# Additional suppressions for deprecated warnings in libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Optionally, you can also suppress warnings from specific libraries (like Keras, if you're using it)
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
import os
import cv2
import torchvision.transforms as T # type: ignore
from sklearn.cluster import KMeans
import matplotlib
import webcolors
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
CORS(app)  # Enable CORS for API access
# ===============================
# Load Models
# ===============================
# Load the NER Model
nlp = spacy.load("trained_ner_model")

# Load the CNN Model
cnn_model = tf.keras.models.load_model("convnext_fashion_model.keras")

# Load the DeepLabV3 Model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()  # Set the model to evaluation mode before inference

# Load Named Colors for Matching
css_colors = matplotlib.colors.CSS4_COLORS
color_names = list(css_colors.keys())
color_values = [matplotlib.colors.to_rgb(css_colors[name]) for name in color_names]

# Load the Fine-Tuned FLAN-T5 Model for Description Generation
model_path = "./fine_tuned_flan_t5"  # Adjust if needed
tokenizer = T5Tokenizer.from_pretrained(model_path)
caption_model = T5ForConditionalGeneration.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

# Build a KDTree for fast lookup
color_tree = KDTree(color_values)
print("models loaded successfully!")

# ===============================
# Attribute Names and Labels
# ===============================
IMG_SIZE = (224, 224)
num_classes_shape = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3]
num_classes_fabric = [8, 8, 8]
num_classes_pattern = [8, 8, 8]

shape_labels = [
    ["sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve", "not long-sleeve", "NA"],
    ["three-point", "medium short", "three-quarter", "long", "NA"],
    ["no", "socks", "leggings", "NA"],
    ["no", "yes", "NA"],
    ["no", "eyeglasses", "sunglasses", "have a glasses in hand or clothes", "NA"],
    ["no", "yes", "NA"],
    ["no", "yes", "NA"],
    ["no", "yes", "NA"],
    ["no", "belt", "have a clothing", "hidden", "NA"],
    ["V-shape", "square", "round", "standing", "lapel", "suspenders", "NA"],
    ["yes", "no", "NA"],
    ["no", "yes", "NA"]
]

fabric_labels = ["denim", "cotton", "leather", "furry", "knitted", "chiffon", "other", "NA"]
pattern_labels = ["floral", "graphic", "striped", "pure color", "lattice", "other", "color block", "NA"]
sleeve_length_map = {
    "sleeveless": "Sleeveless",
    "short-sleeve": "Short Sleeve",
    "medium-sleeve": "Medium Sleeve",
    "long-sleeve": "Long Sleeve",
    "not long-sleeve": "Not Long Sleeve",
    "NA": "NA"
}

lower_clothing_length_map = {
    "three-point": "Three-Point",
    "medium short": "Medium Short",
    "three-quarter": "Three-Quarter",
    "long": "Long",
    "NA": "NA"
}

neckline_map = {
    "V-shape": "V-Shape",
    "square": "Square",
    "round": "Round",
    "standing": "Standing",
    "lapel": "Lapel",
    "suspenders": "Suspenders",
    "NA": "NA"
}

COCO_CLASSES = {15: "person"}


def get_css_color_mapping():
    """Fetch CSS3 color mapping dynamically."""
    return {webcolors.hex_to_rgb(v): k for k, v in webcolors.CSS3_NAMES_TO_HEX.items()}

def find_nearest_color(rgb):
    """Find the nearest CSS3 color name for a given RGB value."""
    try:
        return webcolors.rgb_to_name(rgb), rgb  # Exact match if available
    except ValueError:
        pass  # No exact match, find the closest one

    # Convert input RGB to HEX
    hex_code = webcolors.rgb_to_hex(rgb)

    # Get CSS3 color mappings
    css_colors = get_css_color_mapping()

    # Find the closest color using Euclidean distance
    min_distance = float("inf")
    closest_color = None

    for css_rgb, name in css_colors.items():
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, css_rgb))  # Euclidean distance

        if distance < min_distance:
            min_distance = distance
            closest_color = name

    return closest_color, rgb  # Return the closest color 


# Image Preprocessing
def color_preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = T.Compose([
        T.Resize((512, 512)),  # Ensure the image is resized to 512x512 for DeepLabV3
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0), np.array(image) 

def get_segmentation_mask(image_file):
    input_tensor, original_image = color_preprocess_image(image_file)
    with torch.no_grad():
        output = model(input_tensor)['out']  # Model output should now work fine
    mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    return mask, original_image


def extract_clothing_regions(mask, original_image):
    h, w, _ = original_image.shape
    person_mask = (mask == 15).astype(np.uint8)  # Extract only the "person" mask

    # Resize mask to match the original image size
    person_mask = cv2.resize(person_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Define upper and lower body areas (approximate split)
    upper_body_mask = person_mask[:h//2, :]
    lower_body_mask = person_mask[h//2:, :]

    # Convert masks to uint8
    upper_body_mask = (upper_body_mask * 255).astype(np.uint8)
    lower_body_mask = (lower_body_mask * 255).astype(np.uint8)

    # Apply masks to the original image
    upper_clothing = cv2.bitwise_and(original_image[:h//2], original_image[:h//2], mask=upper_body_mask)
    lower_clothing = cv2.bitwise_and(original_image[h//2:], original_image[h//2:], mask=lower_body_mask)

    return upper_clothing, lower_clothing

def extract_dominant_color(image, k=3):
    pixels = image.reshape((-1, 3))
    pixels = pixels[pixels.sum(axis=1) > 0]  # Remove black pixels

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return tuple(map(int, dominant_color))  # Convert to integer tuple


def process_pure_color(image_file,pat):
    mask, original_image = get_segmentation_mask(image_file)
    upper_clothing, lower_clothing = extract_clothing_regions(mask, original_image)
    # Find the nearest color for the upper and lower clothing
    upper_color = extract_dominant_color(upper_clothing)
    lower_color = extract_dominant_color(lower_clothing)
    upper_color_name, upper_nearest_rgb = find_nearest_color(upper_color)
    lower_color_name, lower_nearest_rgb = find_nearest_color(lower_color)
    
    print("\nColor module:")
    # Get dominant colors
    if(pat =="upper"):
        
        print(f"Upper Clothing Color: {upper_color}")
        print(f"Nearest Upper Clothing Color: {upper_color_name} with RGB value: {upper_nearest_rgb}")
    else:
        
        print(f"Lower Clothing Color: {lower_color}")
        print(f"Nearest Lower Clothing Color: {lower_color_name} with RGB value: {lower_nearest_rgb}")

    
    

def process_description(text):
    """Extracts named entities from the description using the trained NER model."""
    doc = nlp(text)
    ner_output = {}
    print(f"\n Description: {text}")
    for ent in doc.ents:
        ner_output[ent.label_.lower()] = ent.text
    return ner_output


def preprocess_image(image_file):
    """Preprocesses an image uploaded via Flask before CNN model inference."""
    try:
        # Read image from file object as bytes
        image_bytes = image_file.read()

        # Decode image using TensorFlow
        image = tf.image.decode_jpeg(image_bytes, channels=3)

        # Resize image
        image = tf.image.resize(image, IMG_SIZE) / 255.0

        # Add batch dimension
        return tf.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def predict_image_attributes(image_path):
    """Predicts clothing attributes from an image using the CNN model."""
    img_tensor = preprocess_image(image_path)
    
    if img_tensor is None:
        return {}  # Return empty if image processing fails
    
    predictions = cnn_model.predict(img_tensor)
    
    predicted_shape = [shape_labels[i][int(np.argmax(pred))] for i, pred in enumerate(predictions[:len(num_classes_shape)])]
    predicted_fabric = [fabric_labels[int(np.argmax(pred))] for pred in predictions[len(num_classes_shape):len(num_classes_shape)+len(num_classes_fabric)]]
    predicted_pattern = [pattern_labels[int(np.argmax(pred))] for pred in predictions[-len(num_classes_pattern):]]
    cnn_output = {
        "shape_attributes": predicted_shape,
        "fabric_attributes": predicted_fabric,
        "pattern_attributes": predicted_pattern
    }
    return cnn_output
@app.route('/submit', methods=['POST'])
def submit():
    description = request.form.get('description', '')  # Get description from the form
    files = request.files.getlist('images[]')  # Get files sent with the form

    # Print the description and filenames for debugging
    print(f"Received description: {description}")
    for file in files:
        print(f"Received image: {file.filename}")
    ner_output = process_description(description)
    cnn_output = predict_image_attributes(files[0]) if files else {}
    # Trigger the process_pure_color function if "pure color" is detected in pattern attributes
    
    response = {
        "status": "success",
        "extracted_attributes": {
            "ner_output": ner_output,
            "cnn_output": cnn_output
        }
    }

    print("Structured text extractor Output:")

    for key, value in response["extracted_attributes"]["ner_output"].items():
        print(f"- {key}: {value}")
    sleeve_length = sleeve_length_map.get(cnn_output["shape_attributes"][0], "Unknown")
    lower_clothing_length = lower_clothing_length_map.get(cnn_output["shape_attributes"][1], "Unknown")
    neckline = neckline_map.get(cnn_output["shape_attributes"][9], "Unknown")
    # Extract predicted fabric attributes
    fabric_attributes = cnn_output["fabric_attributes"]
    fabric_attributes += ["NA"] * (3 - len(fabric_attributes))  # Ensure at least 3 values
    upper_fabric, lower_fabric, outer_fabric = fabric_attributes[:3]

    # Extract predicted pattern attributes
    pattern_attributes = cnn_output["pattern_attributes"]
    pattern_attributes += ["NA"] * (3 - len(pattern_attributes))  # Ensure at least 3 values
    upper_pattern, lower_pattern, outer_pattern = pattern_attributes[:3]
    print(cnn_output)

    # Print formatted output
    print("\nCNN Output:")

    print(f"- Sleeve Length: {sleeve_length}")
    print(f"- Lower Clothing Length: {lower_clothing_length}")
    print(f"- Neckline: {neckline}")
    print(f"- Upper Fabric: {upper_fabric}")
    print(f"- Lower Fabric: {lower_fabric}")
    print(f"- Outer Fabric: {outer_fabric}")
    print(f"- Upper Pattern: {upper_pattern}")
    print(f"- Lower Pattern: {lower_pattern}")
    print(f"- Outer Pattern: {outer_pattern}")
       
    
    #print color module output
    if upper_pattern=="pure color" :
        process_pure_color(files[0],"upper")
    if(lower_pattern)=="pure color" :
        process_pure_color(files[0],"lower")

# Prepare the final input for the T5 model
    test_input = (
    f"Generate a product description for a product with "
    f"brand: {ner_output['brand']}, "
    f"type: {ner_output['type']}, "
    f"sleeve length: {sleeve_length}, "
    f"lower clothing length: {lower_clothing_length}, "
    f"neckline: {neckline}, "
    f"upper fabric: {upper_fabric}, "
    f"upper pattern: {upper_pattern}, "
    f"lower fabric: {lower_fabric}, "
    f"lower pattern: {lower_pattern}, "
    f"outer fabric: {outer_fabric}, "
    f"outer pattern: {outer_pattern}"
    )
    print(test_input)
    # Tokenize the Input
# ---------------------------
    input_ids = tokenizer.encode(test_input, return_tensors="pt", truncation=True, max_length=128).to(device)

# ---------------------------
# Generate the Caption
# ---------------------------
    outputs = caption_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# Display the Results
# ---------------------------
    print("\nGenerated Product Description:")
    print(generated_text)
    return jsonify({"status": "success", "message": "Description and images received!", "description": generated_text})

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    description = request.json.get('description', '')  # Get description from JSON request
    
    # Placeholder logic for generating a caption
    generated_caption = "a stylish shorts made of leather making a trendy look"
    
    return jsonify({"caption": generated_caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
