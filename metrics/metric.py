# metrics.py
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np
import torch
from tqdm import tqdm

def calculate_rouge_scores(model, processor, test_dataloader, output_file):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    total_rouge1 = 0
    total_rougeL = 0
    total_samples = 0

    for idx, batch in enumerate(tqdm(test_dataloader)):
        documents = batch["document"]
        labels = batch["labels"].to(model.device)
        flattened_patches = batch["flattened_patches"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)

        predictions = processor.batch_decode(outputs.logits.argmax(-1))
        labels = processor.batch_decode(labels)

        for pred, label in zip(predictions, labels):
            eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
            eos_label = label.index("</s>") if "</s>" in label else len(label)

            pred_no_pad = [token for token in pred[:eos_pred] if token != '<pad>']
            label_no_pad = [token for token in label[:eos_label] if token != '<pad>']

            scores = scorer.score(" ".join(pred_no_pad), " ".join(label_no_pad))
            total_rouge1 += scores['rouge1'].fmeasure
            total_rougeL += scores['rougeL'].fmeasure
            total_samples += 1

            for doc, pred, label in zip(documents, predictions, labels):
                eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
                eos_label = label.index("</s>") if "</s>" in label else len(label)

                output_file.write("Document: " + str(doc) + "\n")
                output_file.write("Predictions: " + str(pred[:eos_pred]) + "\n")
                output_file.write("Labels: " + str(label[:eos_label]) + "\n")

    avg_rouge1 = total_rouge1 / total_samples
    avg_rougeL = total_rougeL / total_samples

    return avg_rouge1, avg_rougeL

def calculate_bert_score(model, processor, test_dataloader):
    references = []
    predictions = []

    for batch in tqdm(test_dataloader):
        labels = batch["labels"].to(model.device)
        flattened_patches = batch["flattened_patches"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)

        preds = processor.batch_decode(outputs.logits.argmax(-1))
        refs = processor.batch_decode(labels)

        predictions.extend(preds)
        references.extend(refs)

    P, R, F1 = score(predictions, references, lang="en", verbose=False)

    return np.mean(F1.cpu().numpy())

def calculate_avg_normalized_levenshtein_distance(model, processor, test_dataloader):
    total_distance = 0
    total_samples = 0

    for batch in tqdm(test_dataloader):
        labels = batch["labels"]
        flattened_patches = batch["flattened_patches"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)
        predictions = processor.batch_decode(outputs, skip_special_tokens=True)

        for pred, label in zip(predictions, labels):
            total_distance += levenshtein_distance(pred, label)
            total_samples += 1

    return total_distance / total_samples

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]
