import os
import yaml
from torch.utils.data import Dataset, DataLoader
import torch

import wandb

from models.pix2struct.pix2struct_base_model import Pix2StructModel
from models.pix2struct.pix2struct_dataset import Pix2StructDataset, Pix2StructCollator
from preprocessing.pix2struct.preprocess import pix2struct_preprocess
from models.docOWL.docowl_base_model import DocOWLModel
from models.docOWL.docowl_dataset import DocOWLDataset, DocOWLCollator
from preprocessing.docOWL.preprocess import docowl_preprocess
from sklearn.model_selection import train_test_split

def get_model(config):
    if config["model"]["name"] == "pix2struct":
        return Pix2StructModel(config["model"]["config"]["model_name"], 
                               config["model"]["config"]["processor_name"], 
                               config["model"]["config"]["device"])
    elif config["model"]["name"] == "docowl":
        return DocOWLModel(config["model"]["config"]["docowl_config"]["model_name"], 
                           config["model"]["config"]["docowl_config"]["processor_name"], 
                           config["model"]["config"]["device"])
    else:
        raise ValueError(f"Unknown model name: {config['model']['name']}")

def get_dataset_and_collator(config, model):
    if config["model"]["name"] == "pix2struct":
        dataset = pix2struct_preprocess(config["image_directory"], config["json_directory"])
        train_data, test_data = train_test_split(dataset, test_size=config["validation_size"], random_state=config["random_state"])
        train_dataset = Pix2StructDataset(train_data, model.processor, max_patches=config["max_patches"])
        test_dataset = Pix2StructDataset(test_data, model.processor, max_patches=config["max_patches"])
        collator = Pix2StructCollator(model.processor)
    elif config["model"]["name"] == "docowl":
        dataset = docowl_preprocess(config["image_directory"], config["json_directory"])
        train_data, test_data = train_test_split(dataset, test_size=config["validation_size"], random_state=config["random_state"])
        train_dataset = DocOWLDataset(train_data, model.processor, max_patches=config["max_patches"])
        test_dataset = DocOWLDataset(test_data, model.processor, max_patches=config["max_patches"])
        collator = DocOWLCollator(model.processor)
    else:
        raise ValueError(f"Unknown model name: {config['model']['name']}")
    return train_dataset, test_dataset, collator

if __name__ == "__main__":
    # Read from the config.yaml file
    with open("train_base.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Initialize WandB
    wandb.init(project="your-project-name", config=config)

    # Load the Model
    model = get_model(config)

    # Implement the preprocessing logic
    print("Preprocessing!!")
    train_dataset, test_dataset, collator = get_dataset_and_collator(config, model)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"], collate_fn=collator.pix2struct_collator, num_workers=4)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=config["batch_size"], collate_fn=collator.pix2struct_collator, num_workers=4)

    with open(config["logger_path"], "w") as log_file:
        log_file.write("Logging started...\n")
        log_file.write("Data Sets Used: " + str(os.listdir(config["json_directory"])) + "\n")
        log_file.write(f"Log file used {config}")

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config["learning_rate"])

    for num_epoch in range(config["epochs"]):
        train_loss = model.train(train_dataloader, optimizer)
        with open(config["logger_path"], "a") as log_file:
            validation_loss, rouge1, rougeL = model.evaluate(test_dataloader, log_file)

        # Log everything to wandb
        wandb.log({"Training Loss": train_loss, "Validation Loss": validation_loss, "Average ROUGE-1": rouge1, "Average ROUGE-L": rougeL})

        if num_epoch % 1 == 0:
            model.save(os.path.join(config["weight_path"], f"{config['model']['name']}_epoch_{num_epoch}.pth"))
