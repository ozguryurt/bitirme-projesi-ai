# Epoch sayısı ve batch size parametreleri güncel veri setine göre optimize edilmeli

import json
from collections import defaultdict
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Veri seti
with open("questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open("answers.json", "r", encoding="utf-8") as f:
    answers = json.load(f)

# question_id'ye göre gruplama
answers_by_question = defaultdict(list)
for answer in answers:
    answers_by_question[answer["question_id"]].append(answer["answer_text"])

# soru + cevapları birleştir
data = []
for q in questions:
    q_text = q["question_text"]
    a_text = "\n".join(answers_by_question[q["id"]])
    combined = f"Soru: {q_text}\nCevap: {a_text}"
    data.append(combined)

# hf datasete çevir
dataset = Dataset.from_dict({"text": data})

# tokenizer ve model
model_name = "ytu-ce-cosmos/turkish-gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# tokenizasyon
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data Collator (MLM değil çünkü GPT2)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# eğitim için ayarlar
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    prediction_loss_only=True,
    fp16=False  # GPU'n yoksa True yapma
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# eğitimi yap
trainer.train()

# modeli kaydet
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model başarıyla fine-tune edildi ve ./fine_tuned_model klasörüne kaydedildi.")