from rouge_score import rouge_scorer

def compute_rouge_scores(predictions, labels):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    total_rouge1 = 0
    total_rougeL = 0
    total_samples = 0

    for pred, label in zip(predictions, labels):
        eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
        eos_label = label.index("</s>") if "</s>" in label else len(label)
        pred_no_pad = [token for token in pred[:eos_pred] if token != '<pad>']
        label_no_pad = [token for token in label[:eos_label] if token != '<pad>']
        scores = scorer.score(" ".join(pred_no_pad), " ".join(label_no_pad))
        total_rouge1 += scores['rouge1'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
        total_samples += 1

    avg_rouge1 = total_rouge1 / total_samples
    avg_rougeL = total_rougeL / total_samples
    return avg_rouge1, avg_rougeL