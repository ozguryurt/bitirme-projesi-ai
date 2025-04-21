from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Load the dataset
qa_tr_datasets = load_dataset("ucsahin/TR-Extractive-QA-82K")

# Load model and tokenizer
model_checkpoint = "ucsahin/mT5-base-turkish-qa"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

inference_dataset = qa_tr_datasets["test"].select(range(10))

for input in inference_dataset:
    input_question = "Soru: " + input["question"]
    input_context = "Metin: " + input["context"]

    tokenized_inputs = tokenizer(input_question, input_context, max_length=512, truncation=True, return_tensors="pt")
    outputs = model.generate(input_ids=tokenized_inputs["input_ids"], max_new_tokens=32)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("---")
    print(f"Soru: {input["question"]}")
    print(f"Cevap: {input['answer']}")
    print(f"Model cevabı: {output_text}")
    print("---")