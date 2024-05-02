import os
import yaml
from torch.utils.data import Dataset, DataLoader
import torch
import json
import sys

import wandb

from models.pix2struct.pix2struct_base_model import Pix2StructModel
from models.pix2struct.pix2struct_dataset import Pix2StructDataset, Pix2StructCollator
from preprocessing.pix2struct.preprocess import pix2struct_preprocess
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Read from the config.yaml file
    with open("train_base.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
        image_directory = config.get("image_directory")
        json_directory = config.get("json_directory")

        validation_size = config.get("validation_size")
        random_state = config.get("random_state")

        batch_size = config.get("batch_size")
        learning_rate = config.get("learning_rate")
        epochs = config.get("epochs")

    # Load the Model
    model = Pix2StructModel(config.get("model_name"), config.get("processor_name"), config.get("device"))

    # Implement the preprocessing logic
    print("Preprocessing!!")
    # dataset = pix2struct_preprocess(image_directory, json_directory)
    with open("/data/circulars/CircularsGPT_M/metrics/data/DOCVQA/spdocvqa_data/train_img_Q_A.json", "r") as fp:
        data = json.load(fp)

    print("Keys", data[0].keys())
    dataset = []
    for data_entry in data:
        dataset.append({
            'document': data_entry['document'],
            'question': data_entry['question'],
            'answer': data_entry['answer'][0]
        })

    # Implement the dataset split
    train_data, test_data = train_test_split(dataset, test_size=validation_size, random_state=random_state)

    train_dataset = Pix2StructDataset(train_data, model.processor, max_patches=1024)
    test_dataset = Pix2StructDataset(test_data, model.processor, max_patches=1024)

    collator = Pix2StructCollator(model.processor)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator.pix2struct_collator, num_workers=4)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator.pix2struct_collator, num_workers=4)

    with open(config.get("logger_path"), "w") as log_file:
        log_file.write("Logging started...\n")
        log_file.write("Data Sets Used: " + str(os.listdir(config.get("json_directory"))) + "\n")
        log_file.write(f"Log file used {config}")

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)

    for num_epoch in range(epochs):
        with open(config.get("logger_path"), "a") as log_file:
            validation_loss, rouge1, rougeL = model.evaluate(test_dataloader, log_file)

        # Log everything to wandb
        # wandb.log({"Training Loss": train_loss, "Validation Loss": validation_loss, "Average ROUGE-1": rouge1, "Average ROUGE-L": rougeL})

        if num_epoch % 1 == 0:
            model.save(config.get("weight_path"), num_epoch)
