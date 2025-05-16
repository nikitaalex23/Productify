import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from google.colab import drive


drive.mount('/content/drive')
torch.cuda.empty_cache()
os.environ["WANDB_DISABLED"] = "true"
df = pd.read_csv("product_descriptions_final.csv", encoding="cp1252")
print("Dataset columns:", df.columns)
required_columns = [
    "brand", "type", "sleeve_length", "lower_clothing_length", "neckline",
    "upper_fabric", "upper_pattern", "lower_fabric", "lower_pattern",
    "outer_fabric", "outer_pattern", "description"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' is missing from the dataset!")


#Creating Input Prompt from Feature Columns for the Model

df["input_text"] = (
    "Generate a product description for a product with "
    "brand: " + df["brand"].astype(str) + ", "
    "type: " + df["type"].astype(str) + ", "
    "sleeve length: " + df["sleeve_length"].astype(str) + ", "
    "lower clothing length: " + df["lower_clothing_length"].astype(str) + ", "
    "neckline: " + df["neckline"].astype(str) + ", "
    "upper fabric: " + df["upper_fabric"].astype(str) + ", "
    "upper pattern: " + df["upper_pattern"].astype(str) + ", "
    "lower fabric: " + df["lower_fabric"].astype(str) + ", "
    "lower pattern: " + df["lower_pattern"].astype(str) + ", "
    "outer fabric: " + df["outer_fabric"].astype(str) + ", "
    "outer pattern: " + df["outer_pattern"].astype(str) + "."
)

#Creating a dataset containing the prompt and the target description
dataset = Dataset.from_pandas(df[["input_text", "description"]])

model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

#Tokenizing the Dataset

def tokenize_function(examples):

    inputs = [str(text) for text in examples["input_text"]]
    targets = [str(text) for text in examples["description"]]

    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=6,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none",  # Disable external logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Using same dataset for evaluation here; adjust as needed
)

#Training the Model
trainer.train()
model.save_pretrained("fine_tuned_flan_t5_small")
tokenizer.save_pretrained("fine_tuned_flan_t5_small")
save_path = "/content/drive/MyDrive/finee_tuned_flan_t5_small"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Fine-tuning complete! Model saved in Google Drive at '{save_path}'")

