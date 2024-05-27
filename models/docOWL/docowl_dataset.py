from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor
import torch
import logging

class DocOWLCollator():
    def __init__(self, processor):
        self.processor = processor
        self.logger = logging.getLogger(__name__)

    def docowl_collator(self, batch):
        new_batch = {"input_ids": [], "attention_mask": []}
        texts = [item["text"] for item in batch]
        documents = []
        for item in batch:
            try:
                documents.append(item["document"])
            except KeyError as e:
                self.logger.error(f"KeyError: {e}, item: {item}")
                return None

        text_inputs = self.processor(text=texts, padding="max_length", return_tensors="pt", max_length=128, truncation=True)

        new_batch["input_ids"] = text_inputs.input_ids
        new_batch["attention_mask"] = text_inputs.attention_mask

        for item in batch:
            try:
                new_batch["input_ids"].append(item["input_ids"])
                new_batch["attention_mask"].append(item["attention_mask"])
            except KeyError as e:
                self.logger.error(f"KeyError: {e}, item: {item}")
                return None

        new_batch["input_ids"] = torch.stack(new_batch["input_ids"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        new_batch["document"] = documents
        return new_batch

class DocOWLDataset(Dataset):
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
        processed_data = self.processor(images=image, return_tensors="pt", text=item["text"], max_patches=self.max_patches)
        encoding = {}
        for key in processed_data.keys():
            if key in ['input_ids', 'attention_mask']:
                encoding[key] = processed_data[key].squeeze()
        encoding['text'] = item['text']
        encoding['document'] = item['document']
        return encoding
