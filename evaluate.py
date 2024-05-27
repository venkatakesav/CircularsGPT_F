# import os
# import yaml
# from torch.utils.data import Dataset, DataLoader
# import torch
# import json
# import sys
# import numpy as np

# import wandb

# from models.pix2struct.pix2struct_base_model import Pix2StructModel
# from models.pix2struct.pix2struct_dataset import Pix2StructDataset, Pix2StructCollator
# from preprocessing.pix2struct.preprocess import pix2struct_preprocess
# from sklearn.model_selection import train_test_split

# if __name__ == "__main__":
#     # Read from the config.yaml file
#     with open("evaluate_base.yaml", "r") as yaml_file:
#         config = yaml.safe_load(yaml_file)
#         image_directory = config.get("image_directory")
#         json_directory = config.get("json_directory")

#         validation_size = config.get("validation_size")
#         random_state = config.get("random_state")
#         max_patches = config.get("max_patches")

#         batch_size = config.get("batch_size")
#         learning_rate = config.get("learning_rate")
#         epochs = config.get("epochs")

#     # Load the Model
#     model = Pix2StructModel(config.get("model_name"), config.get("processor_name"), config.get("device"))

#     # Implement the preprocessing logic
#     print("Preprocessing!!")

#     # Get the test dataset
#     dataset = pix2struct_preprocess(image_directory, json_directory, test_flag=True)

#     test_dataset = Pix2StructDataset(dataset, model.processor, max_patches=max_patches)
#     collator = Pix2StructCollator(model.processor)

#     test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator.pix2struct_collator, num_workers=4)

#     with open(config.get("logger_path"), "w") as log_file:
#         log_file.write("Logging started...\n")
#         log_file.write("Data Sets Used: " + str(os.listdir(config.get("json_directory"))) + "\n")
#         log_file.write(f"Log file used {config}")

#     optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)

#     with open(config.get("logger_path"), "a") as log_file:
#         validation_loss, total_rouge1, total_rougeL, total_bert_score, avg_levenshtein_distance = model.evaluate(test_dataloader, log_file)
#         print("Validation Loss", validation_loss)
#         print("Total Rouge 1", total_rouge1)
#         print("Total Rouge L", total_rougeL)
#         print("Total BERT Score", total_bert_score)
#         print("Average Levenstine Distance", avg_levenshtein_distance)

import os
import yaml
import torch
import json
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb

# Import your models and datasets
from models.pix2struct.pix2struct_base_model import Pix2StructModel
from models.pix2struct.pix2struct_dataset import Pix2StructDataset, Pix2StructCollator
from preprocessing.pix2struct.preprocess import pix2struct_preprocess
# Import DocOWL-related modules
from models.docOWL.docowl_base_model import DocOWLModel
from models.docOWL.docowl_dataset import DocOWLDataset
from preprocessing.docowl.preprocess import docowl_preprocess

def get_model(config):
    if config["model"]["name"] == "pix2struct":
        return Pix2StructModel(config["model"]["config"]["model_name"], 
                               config["model"]["config"]["processor_name"], 
                               config["model"]["config"]["device"])
    elif config["model"]["name"] == "docowl":
        return DocOWLModel(config["model"]["config"]["model_name"], 
                           config["model"]["config"]["processor_name"], 
                           config["model"]["config"]["device"])
    else:
        raise ValueError(f"Unknown model name: {config['model']['name']}")

def get_dataset_and_collator(config, model):
    if config["model"]["name"] == "pix2struct":
        dataset = pix2struct_preprocess(config["image_directory"], config["json_directory"], test_flag=True)
        test_dataset = Pix2StructDataset(dataset, model.processor, max_patches=config["max_patches"])
        collator = Pix2StructCollator(model.processor)
    elif config["model"]["name"] == "docowl":
        dataset = docowl_preprocess(config["image_directory"], config["json_directory"], test_flag=True)
        test_dataset = DocOWLDataset(dataset, model.processor, max_patches=config["max_patches"])
        collator = DocOWLCollator(model.processor)
    else:
        raise ValueError(f"Unknown model name: {config['model']['name']}")
    return test_dataset, collator

if __name__ == "__main__":
    # Read from the config.yaml file
    with open("evaluate_base.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Load the Model
    model = get_model(config)

    # Implement the preprocessing logic
    print("Preprocessing!!")

    # Get the test dataset and collator
    test_dataset, collator = get_dataset_and_collator(config, model)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=config["batch_size"], collate_fn=collator.collate_fn, num_workers=4)

    with open(config["logger_path"], "w") as log_file:
        log_file.write("Logging started...\n")
        log_file.write("Data Sets Used: " + str(os.listdir(config["json_directory"])) + "\n")
        log_file.write(f"Log file used {config}")

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config["learning_rate"])

    with open(config["logger_path"], "a") as log_file:
        validation_loss, total_rouge1, total_rougeL, total_bert_score, avg_levenshtein_distance = model.evaluate(test_dataloader, log_file)
        print("Validation Loss", validation_loss)
        print("Total Rouge 1", total_rouge1)
        print("Total Rouge L", total_rougeL)
        print("Total BERT Score", total_bert_score)
        print("Average Levenstine Distance", avg_levenshtein_distance)
