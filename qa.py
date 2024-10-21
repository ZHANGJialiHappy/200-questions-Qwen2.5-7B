from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv

# Set device to GPU if available, otherwise CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define model name and token for authentication
lm = 'vivo-ai/BlueLM-7B-Base'
token = 'Token'  # Replace 'YOUR_HF_TOKEN_HERE' with your Hugging Face token

# Load model and tokenizer with authentication token
lang_model = AutoModelForCausalLM.from_pretrained(lm, token=token,trust_remote_code=True)
lang_model.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(lm, token=token, use_fast=False, trust_remote_code=True)

# Load questions from CSV file
questions = open('questions.csv').readlines()

# Open a CSV file to write answers
with open('BlueLM-7B-Base.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header to CSV file
    csvwriter.writerow(['Question', 'Answer'])

    prefixes = ['']
    postfixes = ['']

    # Iterate over questions and generate answers
    for prefix, postfix in zip(prefixes, postfixes):
        for question in questions:
            question = prefix + ' ' + question.strip() + ' ' + postfix
            # Tokenize question and move to device (GPU/CPU)
            tokked = tokenizer(question.strip(), return_tensors='pt', truncation=True, padding=True)['input_ids']
            tokked = tokked.to(DEVICE)

            # Use no_grad to avoid storing gradients
            with torch.no_grad():
                # Generate answer from the model
                generated_ids = lang_model.generate(tokked, max_new_tokens=200)
                tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Print question and answer
            print(question)
            print(' '.join(tokens))
            print()

            # Write question and answer to the CSV file
            csvwriter.writerow([question, ' '.join(tokens)])

            # Free GPU memory after each question
            torch.cuda.empty_cache()
