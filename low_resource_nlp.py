# low-resource-nlp/low_resource_nlp.py

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer for low-resource languages
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

# Example input
text = "This is an example sentence."

# Tokenize and classify
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
print(outputs)
