import torch
from transformers import DistilBertTokenizer
import mlflow.pytorch

# Define model path (replace 'model_uri' with the URI where your model is stored in MLflow)
# Example: if it's in the latest run of an experiment, use 'models:/mlm-distilbert-model/Production' or a similar path.
model_uri = "models:/DistilBERT-MLM-Experiment/Production"  # Adjust version or use 'Production' if registered

# Load model from MLflow
loaded_model = mlflow.pytorch.load_model(model_uri)

# Load tokenizer (assuming it's saved as an artifact in the model's MLflow path)
tokenizer = DistilBertTokenizer.from_pretrained("tokenizer")  # Adjust path if needed

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)
loaded_model.eval()  # Set model to evaluation mode

# Define a function to make predictions
def predict_masked_text(text):
    # Tokenize input text with a masked token (e.g., '[MASK]')
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict masked token
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        predictions = outputs.logits

    # Get the index of the [MASK] token
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    mask_token_logits = predictions[0, mask_token_index, :]

    # Decode the prediction of the masked token
    predicted_token_id = mask_token_logits.argmax(dim=-1).item()
    predicted_token = tokenizer.decode([predicted_token_id])

    # Replace [MASK] token with predicted token in the original text
    result_text = text.replace(tokenizer.mask_token, predicted_token)
    return result_text

# Example usage
text = "The capital of Belgium is [MASK]."
predicted_text = predict_masked_text(text)
print("Predicted text:", predicted_text)