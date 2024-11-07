import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns

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
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# MLflow experiment setup
mlflow.set_experiment("DistilBERT-MLM-Experiment")

# Start MLflow run
with mlflow.start_run():
    # Log training arguments as parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("per_device_train_batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
    mlflow.log_param("mlm_probability", 0.15)

    # Start training and evaluation
    trainer.train()
    
    model_save_path = "./huggingface_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Log the model
    mlflow.pytorch.log_model(model, "mlm-distilbert-model")

    # Optional: Log the tokenizer files as artifacts
    # tokenizer.save_pretrained("tokenizer")
    mlflow.log_artifacts("tokenizer", artifact_path="tokenizer")

    # Log final metrics if needed
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        mlflow.log_metric(key, value)
        
    # Plot training loss
    training_loss = [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
    epochs = [entry['epoch'] for entry in trainer.state.log_history if 'loss' in entry]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=training_loss, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("training_loss.png")
    plt.show()
    