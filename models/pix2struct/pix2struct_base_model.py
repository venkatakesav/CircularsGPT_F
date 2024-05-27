# models/pix2struct/model.py
from models import BaseModel
from transformers import Pix2StructForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import torch
from metrics.metric import calculate_rouge_scores, calculate_bert_score, calculate_avg_normalized_levenshtein_distance

class Pix2StructModel(BaseModel):
    def __init__(self, model_name, processor_name, device):
        self.model_name = model_name
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(processor_name)

        self.device = device
        self.model.to(device)

    def train(self, train_dataloader, optimizer):
        # Implement the training logic here
        self.model.train()
        train_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            labels = batch["labels"].to(self.device)
            flattened_patches = batch["flattened_patches"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
        print("Training Loss", train_loss/len(train_dataloader))
        return train_loss/len(train_dataloader)

    def evaluate(self, test_dataloader, output_file):
        output_file.write("For The Next Round of Evaluation")
        # Implement the evaluation logic here
        with torch.no_grad():
            validation_loss = 0
            total_samples = 0
            total_rouge1, total_rougeL = calculate_rouge_scores(self.model, self.processor, test_dataloader, output_file)
            total_bert_score = calculate_bert_score(self.model, self.processor, test_dataloader)
            avg_levenshtein_distance = calculate_avg_normalized_levenshtein_distance(self.model, self.processor, test_dataloader)

            validation_loss /= len(test_dataloader)

        output_file.write("=============================================\n")
        output_file.write(f"Validation loss: {validation_loss}, Average Rouge 1: {total_rouge1}, Average Rouge L: {total_rougeL}, Average BERT Score: {total_bert_score}, Average Levenshtein Distance: {avg_levenshtein_distance}")
        return validation_loss, total_rouge1, total_rougeL, total_bert_score, avg_levenshtein_distance

    def predict(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        predictions = self.model.generate(**inputs)
        return self.processor.decode(predictions[0], skip_special_tokens=True)

    def save(self, path, num_epoch):
        self.model.save_pretrained(f"{path}/{self.model_name}_2048_{num_epoch}")
