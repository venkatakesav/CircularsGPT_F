import json
from tqdm import tqdm

with open("/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/image_ocr.json", "r") as file:
    image_ocr_dict = json.load(file)

## adding ques and answers of validation dataset to image_ocr
with open("/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/val_v1.0_withQT.json", "r") as file:
    ocr_dict = json.load(file)

print(len(ocr_dict['data']))

val_images = []

# Use tqdm to add a progress bar to the loop
for data in tqdm(ocr_dict['data']):
    temp = {}
    image_id = f"{data['ucsf_document_id']}_{data['ucsf_document_page_no']}"
    temp['document'] = image_ocr_dict[image_id]['image_path']
    temp['question'] = data["question"]
    temp["answer"] = data["answers"]
    temp["question_types"] = data["question_types"]
    val_images.append(temp)

save_dict_path = "/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/val_img_Q_A.json"
with open(save_dict_path, 'w') as json_file:
    json.dump(val_images, json_file, indent=4)

## adding ques and answers of train dataset to image_ocr
with open("/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/train_v1.0_withQT.json", "r") as file:
    ocr_dict = json.load(file)

train_images=[]
for data in tqdm(ocr_dict['data']):
    temp={}
    image_id=f"{data['ucsf_document_id']}_{data['ucsf_document_page_no']}"
    temp['document']=image_ocr_dict[image_id]['image_path']
    temp['question']= data["question"]
    temp["answer"] = data["answers"]
    temp["question_types"]=data["question_types"]
    train_images.append(temp)

save_dict_path = "/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/train_img_Q_A.json"
with open(save_dict_path, 'w') as json_file:
    json.dump(train_images, json_file, indent=4)

# save_dict_path = "/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/train_img_Q_A.json"
# with open(save_dict_path, 'r') as json_file:
#     train_images = json.load(json_file)
# print(len(train_images))

# save_dict_path = "/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/val_img_Q_A.json"

# with open(save_dict_path, 'r') as json_file:
#     val_images = json.load(json_file)
# print(len(val_images))

# print(val_images[5])


