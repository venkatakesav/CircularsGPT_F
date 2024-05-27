import json
import os
from tqdm import tqdm

def preprocess_text(text):
    # Remove '\\n' and space before '?'
    text = text.replace('\\n', '').replace(' ?', '?')
    # Replace '\\\"' with single quote
    text = text.replace('\\\"', "'")
    return text

def pix2struct_preprocess(image_directory, json_directory, test_flag=False):
    data_dir = './Dataset/jsons'
    jsons = ['first_page_batch_2.json', 'first_page_batch_4.json', 'last_page_batch_4.json', 'middle_page_batch_2.json']
    final_dataset = []

    json_files = os.listdir(json_directory)
    print("JSON Files:", json_files)

    for json_file in json_files:
        if json_file == "test_batch_1.json":
            print("Skipping File:", json_file)
            continue

        with open(os.path.join(json_directory, json_file)) as f:
            train_data = json.load(f)

        # Batch Number
        batch_no = json_file.split(".")[0][-1]

        temp_array = []
        for data in train_data:
            hindi_flag = False
            for entry in data[0]["question_answer_pairs"]:
            # for entry in data["question_answer_pairs"]:
                temp = {}
                if test_flag:
                    temp["document"] = f"{image_directory}/{data[0]['file_name']}"
                else:
                    temp["document"] = f"{image_directory}{json_directory.split('.')[-1].split('/')[-1]}/Batch_{batch_no}/{json_file.split('.')[0]}/{data[0]['file_name']}"
                temp["question"] = preprocess_text(entry["question"])
                temp["answer"] = preprocess_text(entry["answer"])

                # Check for Hindi text
                if any(ord(char) > 128 for char in entry["question"]) or any(ord(char) > 128 for char in entry["answer"]):
                    hindi_flag = True

                if "claude-3-haiku" not in temp["answer"]:
                    temp_array.append(temp)

            if hindi_flag:
                break

        if not hindi_flag:
            question_set = set()
            answer_set = set()
            unique_array = []
            for entry in temp_array:
                if entry["question"] not in question_set and entry["answer"] not in answer_set:
                    question_set.add(entry["question"])
                    answer_set.add(entry["answer"])
                    unique_array.append(entry)
            final_dataset.extend(unique_array)

    print("Final Dataset Length:", len(final_dataset))
    with open('final_dataset_pix2struct.json', 'w') as f:
        json.dump(final_dataset, f)

    return final_dataset
