import os
import math
import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set device for model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer initialization for DistilBERT
model_name = "distilbert-base-uncased"
model = DistilBertForMaskedLM.from_pretrained(model_name)
model.to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Load and prepare dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./mlm-model",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# Start training and evaluation
trainer.train()



model_save_path = "./huggingface_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Extract training loss and validation perplexity
training_loss = trainer.state.log_history
print(training_loss)
# Extract loss, grad_norm, and epoch
loss_values = []
grad_norm_values = []
epochs = []

for entry in training_loss:
    if 'loss' in entry and 'grad_norm' in entry and 'epoch' in entry:
        loss_values.append(entry['loss'])
        grad_norm_values.append(entry['grad_norm'])
        epochs.append(entry['epoch'])

# Convert epochs to numpy array for polyfit
epochs_np = np.array(epochs)

# ----- Plot Loss Over Epochs with Trend Line -----
plt.figure(figsize=(10, 6))
sns.lineplot(x=epochs, y=loss_values, marker='o', label='Loss')

# Calculate trend line
loss_trend = np.polyfit(epochs_np, loss_values, 1)
loss_trend_line = np.polyval(loss_trend, epochs_np)
plt.plot(epochs_np, loss_trend_line, color='red', label='Trend Line')

plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join('images', "loss_with_trend.png"))
plt.show()

# ----- Plot Grad Norm Over Epochs with Trend Line -----
plt.figure(figsize=(10, 6))
sns.lineplot(x=epochs, y=grad_norm_values, marker='o', label='Grad Norm', color='orange')

# Calculate trend line
grad_trend = np.polyfit(epochs_np, grad_norm_values, 1)
grad_trend_line = np.polyval(grad_trend, epochs_np)
plt.plot(epochs_np, grad_trend_line, color='red', label='Trend Line')

plt.title('Grad Norm Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Grad Norm')
plt.legend()
plt.savefig(os.path.join('images', "grad_norm_with_trend.png"))
plt.show()