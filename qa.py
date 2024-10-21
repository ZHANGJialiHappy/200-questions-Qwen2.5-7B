from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

lm = 'Qwen/Qwen2.5-7B'
token = 'Not_Now'
lang_model = AutoModelForCausalLM.from_pretrained(lm, token=token)
lang_model.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(lm, token=token, use_fast=False)

questions = open('questions.csv').readlines()

with open('Qwen2.5_7B_answer.txt', 'w') as answer_file:
    prefixes = ['']
    postfixes = ['']

    for prefix, postfix in zip(prefixes, postfixes):
        for question in questions:
            question = prefix + ' ' + question.strip() + ' ' + postfix
            tokked = tokenizer(question.strip(), return_tensors='pt', truncation=True, padding=True)['input_ids']
            tokked = tokked.to(DEVICE)
            generated_ids = lang_model.generate(tokked, max_new_tokens=200)
            tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # 打印问题和答案
            print(question)
            print(' '.join(tokens))
            print()

            # 将答案写入文件
            # answer_file.write('Question: ' + question + '\n')
            answer_file.write('Answer: ' + ' '.join(tokens) + '\n\n')
            

