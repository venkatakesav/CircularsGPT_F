import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

image_ocr_path = '/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/image_ocr.json'
val_q_a_path = '/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/val_img_Q_A.json'
output_file = 'predictions_mistral_7b.json'

with open(image_ocr_path) as f:
    image_ocr = json.load(f)

with open(val_q_a_path) as f:
    val_q_a = json.load(f)

device = "cuda:1"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

model.to(device)

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

    messages = [
        {"role": "user", "content": f"{input_text}"},
    ]

    encodeds = tokenizer.apply_chat_template(messages,tokenize=True, add_generation_prompt=False, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    predicted = tokenizer.batch_decode(generated_ids)
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
