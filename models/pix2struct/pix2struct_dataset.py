from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor
import torch
import logging

class Pix2StructCollator():
    def __init__(self, processor):
        self.processor_base = processor
        self.logger = logging.getLogger(__name__)

    def pix2struct_collator(self, batch):
        new_batch = {"flattened_patches":[], "attention_mask":[]}
        texts = [item["answer"] for item in batch]
        documents = []
        for item in batch:
            try:
                documents.append(item["document"])
            except KeyError as e:
                self.logger.error(f"KeyError: {e}, item: {item}")
                return None
        
        text_inputs = self.processor_base(text=texts, padding="max_length", return_tensors="pt", max_length=128, truncation=True)
        
        new_batch["labels"] = text_inputs.input_ids
        
        for item in batch:
            try:
                new_batch["flattened_patches"].append(item["flattened_patches"])
                new_batch["attention_mask"].append(item["attention_mask"])
            except KeyError as e:
                self.logger.error(f"KeyError: {e}, item: {item}")
                return None
        
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"]) 
        new_batch["document"] = documents
        return new_batch

class Pix2StructDataset(Dataset):
    def __init__(self, data, processor, max_patches):
        self.data = data
        self.processor = processor
        self.max_patches = max_patches
        self.logger = logging.getLogger(__name__)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item['document'])
        except Exception as e:
            self.logger.error(f"Error opening image: {e}, item: {item}")
            return None
        processed_data = self.processor(images=image, return_tensors="pt", text=item["question"], max_patches=self.max_patches)
        encoding = {}
        for key in processed_data.keys():
            if key in ['flattened_patches', 'attention_mask']:
                encoding[key] = processed_data[key].squeeze()
        encoding['answer'] = item['answer']
        encoding['question'] = item['question']
        encoding['document'] = item['document']
        return encoding
