# models/pix2struct/model.py
from models import BaseModel
from transformers import Pix2StructForConditionalGeneration, AutoProcessor,Pix2StructProcessor
from tqdm import tqdm
import torch
from rouge_score import rouge_scorer

class Pix2StructModel(BaseModel):
    def __init__(self, model_name, processor_name, device):
        self.model_name = model_name.split("/")[-1]
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
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        output_file.write("For The Next Round of Evaluation")
        # Implement the evaluation logic here
        with torch.no_grad():
            validation_loss = 0
            total_rouge1 = 0
            total_rougeL = 0
            total_samples = 0
            total_anls = 0
            for idx_1, batch_1 in enumerate(tqdm(test_dataloader)):
                documents = batch_1["document"]
                labels = batch_1["labels"].to(self.device)
                flattened_patches = batch_1["flattened_patches"].to(self.device)
                attention_mask = batch_1["attention_mask"].to(self.device)

                outputs = self.model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                validation_loss += loss.item()

                # Decode predictions and labels
                predictions = self.processor.batch_decode(outputs.logits.argmax(-1))
                labels = self.processor.batch_decode(labels)

                # Calculate ROUGE scores
                for pred, label in zip(predictions, labels):
                    eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
                    eos_label = label.index("</s>") if "</s>" in label else len(label)

                    pred_no_pad = [token for token in pred[:eos_pred] if token != '<pad>']
                    label_no_pad = [token for token in label[:eos_label] if token != '<pad>']

                    scores = scorer.score(" ".join(pred_no_pad), " ".join(label_no_pad))
                    total_rouge1 += scores['rouge1'].fmeasure
                    total_rougeL += scores['rougeL'].fmeasure
                    total_samples += 1

                # Write predictions and labels to file
                for doc, pred, label in zip(documents, predictions, labels):
                    eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
                    eos_label = label.index("</s>") if "</s>" in label else len(label)

                    output_file.write("Document: " + str(doc) + "\n")
                    output_file.write("Predictions: " + str(pred[:eos_pred]) + "\n")
                    output_file.write("Labels: " + str(label[:eos_label]) + "\n")

            avg_rouge1 = total_rouge1 / total_samples
            avg_rougeL = total_rougeL / total_samples

        output_file.write("=============================================\n")
        output_file.write(f"Validation loss: {validation_loss/len(test_dataloader)}, Average Rouge 1: {avg_rouge1}, Average Rouge L: {avg_rougeL}")
        return validation_loss/len(test_dataloader), avg_rouge1, avg_rougeL

    def predict(self, image,question):
        # Implement the prediction logic here
        # model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-ai2d-base").to("cuda")
        # processor = Pix2StructProcessor.from_pretrained("google/pix2struct-ai2d-base")

        # question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"

        inputs = self.processor(images=image, text=question, return_tensors="pt").to("cuda")

        predictions = self.model.generate(**inputs)
        print(self.processor.decode(predictions[0], skip_special_tokens=True))


        pass

    def save(self, path, num_epoch):
        self.model.save_pretrained(f"{path}/{self.model_name}_half_{num_epoch}")