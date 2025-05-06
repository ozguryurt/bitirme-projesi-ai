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
import random

# 1. Soru ve Cevapları Yükle
with open("questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open("answers.json", "r", encoding="utf-8") as f:
    answers = json.load(f)

# 2. Cevapları question_id'ye göre grupla
answers_by_question = defaultdict(list)
for answer in answers:
    answers_by_question[answer["question_id"]].append(answer["answer_text"])

# 3. Soru + En İyi Cevabı birleştir (her soru için sadece bir cevap)
data = []
for q in questions:
    q_text = q["question_text"]
    
    # Eğer birden fazla cevap varsa, en uzun ve kapsamlı olanı seç
    if answers_by_question[q["id"]]:
        # En uzun cevabı seç (genellikle daha kapsamlı olur)
        a_text = max(answers_by_question[q["id"]], key=len)
        combined = f"Soru: {q_text}\nCevap: {a_text}"
        data.append(combined)

# Veri setini karıştır
random.shuffle(data)

# 4. Hugging Face Dataset'e çevir
dataset = Dataset.from_dict({"text": data})

# 5. Tokenizer ve Model Yükle
model_name = "ytu-ce-cosmos/turkish-gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# 6. Tokenizasyon
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 7. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 8. Eğitim Ayarları - Eski sürüm için uyumlu parametreler
try:
    # GPU kontrolü
    import torch
    fp16_option = torch.cuda.is_available()
except:
    fp16_option = False

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    prediction_loss_only=True,
    fp16=fp16_option,  # GPU varsa True, yoksa False
    # Eski versiyonlarla uyumlu olmayan parametreler kaldırıldı
)

# 9. Trainer oluştur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 10. Eğitimi Başlat
trainer.train()

# 11. Eğitilen modeli kaydet
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model başarıyla fine-tune edildi ve ./fine_tuned_model klasörüne kaydedildi.")