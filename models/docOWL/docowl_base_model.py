from models import BaseModel
from tqdm import tqdm
import torch
from metrics.metric import calculate_avg_normalized_levenshtein_distance, calculate_bert_score, calculate_rouge_scores
from models.docOWL.mplug_docowl.inference import DocOwlInfer

class DocOWLModel(BaseModel):
    def __init__(self, model_name, processor_name, device):
        self.model_name = model_name
        self.device = device
        self.model = DocOwlInfer(ckpt_path=model_name)

    def evaluate(self, test_dataset, output_file):
        output_file.write("For The Next Round of Evaluation")

    def predict(self, image, question):
        answer = self.model.inference(image, question)
        return answer
    
    def save(self, path, num_epoch):
        self.model.save_pretrained(f"{path}/{self.model_name}_2048_{num_epoch}")