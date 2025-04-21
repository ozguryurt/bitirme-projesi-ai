# train.py

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri setini yükle
dataset = load_dataset("toughdata/quora-question-answer-dataset")

# Yüzdelik veri kullanımı (%10 örnek)
fraction = 0.3
subset_len = int(len(dataset["train"]) * fraction)
dataset["train"] = dataset["train"].select(range(subset_len))

# Tokenizer ve model
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Preprocessing fonksiyonu: question → input, answer → label
"""
def preprocess(example):
    question = example["question"]
    answer = example["answer"]
    if not question or not answer:
        return {}  # boş kayıtları atla
    inputs = tokenizer(question, truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(answer, truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = labels["input_ids"]
    return inputs
"""
def preprocess(example):
    question = "question: " + example["question"]
    answer = example["answer"]
    if not question or not answer:
        return {}
    inputs = tokenizer(question, truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(answer, truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = labels["input_ids"]
    return inputs

# Veriyi dönüştür
train_data = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
train_data = train_data.filter(lambda x: "labels" in x)  # Boş/bozuk verileri temizle

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./quora_model",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_dir="./logs",
    save_total_limit=1,
    save_steps=500,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Eğitimi başlat
trainer.train()

# Modeli kaydet
model.save_pretrained("./quora_model")
tokenizer.save_pretrained("./quora_model")