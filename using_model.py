import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

# Model and tokenizer initialization for DistilBERT
model_name = "distilbert-base-uncased"

# Load model and tokenizer from Hugging Face
model = DistilBertForMaskedLM.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Define a function to make predictions
def predict_masked_text(text):
    # Tokenize input text with a masked token (e.g., '[MASK]')
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict masked token
    with torch.no_grad():
        outputs = model(**inputs)
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
text = "Belgium is a [MASK]."
predicted_text = predict_masked_text(text)
print("Predicted text:", predicted_text)
