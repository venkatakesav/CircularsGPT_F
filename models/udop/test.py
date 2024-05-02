from transformers import AutoProcessor, UdopForConditionalGeneration
from datasets import load_dataset

# load model and processor
# in this case, we already have performed OCR ourselves
# so we initialize the processor with `apply_ocr=False`
processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large")

# load an example image, along with the words and coordinates
# which were extracted using an OCR engine
dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
question = "Question answering. What is the date on the form?"

# prepare everything for the model
encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")

# autoregressive generation
predicted_ids = model.generate(**encoding)
print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])