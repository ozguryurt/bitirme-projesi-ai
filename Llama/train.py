import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# 1. Model ve tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. 8-bit GPU destekli model yükleme
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

# 3. LoRA konfigürasyonu
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

# 4. CSV dosyasını okuma ve dataset oluşturma
df = pd.read_csv("soru_cevap.csv")
df["text"] = df.apply(lambda row: f"Soru: {row['soru']}\nCevap: {row['cevap']}", axis=1)
dataset = Dataset.from_pandas(df[["text"]])

# 5. Tokenize etme
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize)

# 6. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./tinyllama-qa-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none"
)

# 8. Trainer ve eğitim
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# 9. Modeli kaydet
model.save_pretrained("tinyllama-qa-lora")
tokenizer.save_pretrained("tinyllama-qa-lora")

print("✅ Eğitim tamamlandı ve model tinyllama-qa-lora klasörüne kaydedildi.")