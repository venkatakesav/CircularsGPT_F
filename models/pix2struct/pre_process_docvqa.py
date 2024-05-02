import json
import os
import glob
from tqdm import tqdm


## getting image_ocr for entire dataset
def get_image_paths(folder_path):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_paths

folder_path = "/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_images/"
image_paths = get_image_paths(folder_path)
print(len(image_paths))
image_ocr_dict = {}

for path in tqdm(image_paths):
    # print(path)
    ocr_path = path.split('/')
    ocr_path[-1] = ocr_path[-1][:-3] + "json"
    ocr_path[-2] = "spdocvqa_ocr"
    ocr_path = "/".join(ocr_path)

    with open(ocr_path, "r") as file:
        ocr_dict = json.load(file)
    text = []
    for line in ocr_dict["recognitionResults"][0]['lines']: 
        text.append(line['text'])
    image_id = path.split('/')[-1][:-4]
    image_ocr_dict[image_id] = {'image_path': path, 'ocr_text': text}  # text is a list of each line

save_dict_path = "/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/image_ocr.json"
with open(save_dict_path, 'w') as json_file:
    json.dump(image_ocr_dict, json_file, indent=4)
