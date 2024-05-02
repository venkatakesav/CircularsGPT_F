from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor,AutoProcessor
import json

from rouge import Rouge

def calculate_rouge_score(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]  # Return the first score (ROUGE-N, ROUGE-L, ROUGE-W)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base").to('cuda:0')
processor = AutoProcessor.from_pretrained("google/pix2struct-docvqa-base")

# model = Pix2StructForConditionalGeneration.from_pretrained("/data/circulars/CircularsGPT_M/weights/pix2struct-large_0_half_10").to('cuda:0')
# processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")

# with open("/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/val_img_Q_A.json","r")as f:
#     train_image_qa = json.load(f)
with open("/data/circulars/CircularsGPT_M/final_dataset_pix2struct.json","r")as f:
    train_image_qa = json.load(f)


for index in range(100,300,30):

    image = Image.open(train_image_qa[index]['document'])
    question = train_image_qa[index]['question']
    inputs = processor(images=image, text=question, return_tensors="pt").to('cuda:0')
    predictions = model.generate(**inputs)
    print("question:-",question)
    answer1=processor.decode(predictions[0], skip_special_tokens=True)
    print("generated_ans=",answer1)
    answer_gt =train_image_qa[index]['answer']
    print("ground truth=",answer_gt)
    rouge_scores = calculate_rouge_score(answer1, answer_gt[0])
    print(rouge_scores)
    print(train_image_qa[index]['document'])
    print("="*20)

