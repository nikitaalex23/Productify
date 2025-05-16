import spacy
from spacy.training.example import Example
import random
import json

# Load blank English model in spaCy
nlp = spacy.blank("en")

# Add the Named Entity Recognizer (NER) pipeline if it's not in the pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Load the labeled training data
with open("final_dataset.json", "r") as f:
    training_data = json.load(f)

# Add labels to the NER model based on training data
for item in training_data:
    for entity in item['entities']:
        ner.add_label(entity[2])

# Training parameters
n_iter = 20
optimizer = nlp.initialize()

# Train the model
for i in range(n_iter):
    print(f"Starting iteration {i+1}")
    random.shuffle(training_data)
    losses = {}
    for item in training_data:
        text = item['text']
        annotations = {"entities": item['entities']}
        example = Example.from_dict(nlp.make_doc(text), annotations)
        
        # Update the model with the example
        nlp.update([example], losses=losses, drop=0.35, sgd=optimizer)
    print(f"Losses at iteration {i+1}: {losses}")

# Save the trained model
nlp.to_disk("trained_ner_model")
print("Training complete and model saved as 'trained_ner_model'.")