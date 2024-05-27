import json

ocr_path = "./predictions_gemma_2b.json"

with open(ocr_path, "r") as f:
    ocr_data = json.load(f)

pred_gemma=[]
truth_gemma=[]
for i in range(len(ocr_data)):
    # given index of string "based on above text answer the following question" in a variable x
    predicted = ocr_data[i]['predicted']
    x = predicted.index("based on above text answer the following question")
    predicted = predicted[x:]
    x = predicted.index("?")
    predicted = predicted[x+3:]
    if "<eos>" in predicted:
        x = predicted.index("<eos>")
        predicted=predicted[:x]
    pred_gemma.append(predicted)
    truth_gemma.append(ocr_data[i]['ground_truth'][-1])

ocr_path = "./predictions_mistral_7b.json"

with open(ocr_path, "r") as f:
    ocr_data = json.load(f)

pred_mistral=[]
truth_mistral=[]
for i in range(len(ocr_data)):
    # given index of string "based on above text answer the following question" in a variable x
    predicted = ocr_data[i]['predicted'][0]
    x = predicted.index("based on above text answer the following question")
    predicted = predicted[x:]
    x = predicted.index("[/INST]")
    predicted = predicted[x+7:-4]
    if "</s>" in predicted:
        x = predicted.index("</s>")
        predicted=predicted[:x]
    pred_mistral.append(predicted)
    truth_mistral.append(ocr_data[i]['ground_truth'][-1])


#calcuate bert score, anls score and rouge score between predicted and ground truth and report average of all scores for each model

from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Calculate BERTScore
def calculate_bert_score(pred_list, ref_list):
    _, _, bert_scores = bert_score(pred_list, ref_list, lang='en', verbose=True)
    return bert_scores.mean().item()

# Calculate BLEU Score
def calculate_bleu_score(pred_list, ref_list):
    return corpus_bleu([[ref] for ref in ref_list], pred_list)

# Calculate Meteor Score
def calculate_meteor_score(pred_list, ref_list):
    meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(pred_list, ref_list)]
    return sum(meteor_scores) / len(meteor_scores)

# Calculate Rouge Score
def calculate_rouge_score(pred_list, ref_list):
    rouge = Rouge()
    scores = rouge.get_scores(pred_list, ref_list, avg=True)
    return {key: value['f'] for key, value in scores.items()}

# Calculate scores for Gemma and Mistral
for model_name, pred_list,ref_list in [("Gemma", pred_gemma,truth_gemma), ("Mistral", pred_mistral,truth_mistral)]:
    
    bert_avg = calculate_bert_score(pred_list, ref_list)
    bleu_avg = calculate_bleu_score(pred_list, ref_list)
    rouge_avg = calculate_rouge_score(pred_list, ref_list)
    
    print(f"Model: {model_name}")
    print(f"BERT Score: {bert_avg:.4f}")
    print(f"BLEU Score: {bleu_avg:.4f}")
    print(f"Rouge Score: {rouge_avg}")
    print()
