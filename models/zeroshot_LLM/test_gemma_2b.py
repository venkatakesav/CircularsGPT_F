import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

image_ocr_path = '/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/image_ocr.json'
val_q_a_path = '/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/val_img_Q_A.json'
output_file = 'predictions_gemma_2b.json'

with open(image_ocr_path) as f:
    image_ocr = json.load(f)

with open(val_q_a_path) as f:
    val_q_a = json.load(f)

device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.float16,
    revision="float16",
).to(device)

data = []  # List to store results
cur_list=[]
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        json.dump(cur_list, f, indent=4)  # Indent for readability
a=0

for ques in tqdm(val_q_a):
    image_id = ques['document'].split('/')[-1].split('.')[0]
    ocr = '\n'.join(image_ocr[image_id]["ocr_text"])
    question = ques['question']
    ground_truth = ques['answer']

    input_text = f"{ocr}\n\n based on above text answer the following question\n\n{question}"
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(**input_ids, max_length=1000)
    predicted = tokenizer.decode(outputs[0])

    # Create a dictionary for each question-answer pair
    result = {
        "image_path": ques['document'],
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "ocr": ocr,
    }
    data.append(result)
    if a>3:
        with open(output_file, 'r') as f:
            cur_list = json.load(f)  # Indent for readability
        cur_list.extend(data)
        with open(output_file, 'w') as f:
            json.dump(cur_list, f, indent=4)  # Indent for readability
        data=[]

    a+=1

with open(output_file, 'r') as f:
    cur_list = json.load(f)
cur_list.extend(data)

with open(output_file, 'w') as f:
    json.dump(cur_list, f, indent=4)

print(f"Results saved to: {output_file}")
