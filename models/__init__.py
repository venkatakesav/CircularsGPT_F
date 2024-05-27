# models/__init__.py
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self, train_dataloader: DataLoader):
        pass

    @abstractmethod
    def evaluate(self, eval_dataloader: DataLoader):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
        self.to(self.device)