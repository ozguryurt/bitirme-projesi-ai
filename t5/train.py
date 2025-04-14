import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os

# --- Ayarlar ---
MODEL_NAME = "google-t5/t5-small"
CSV_PATH = "soru_cevap.csv"
OUTPUT_DIR = "./t5_soru_cevap_model" # Eğitilmiş modelin kaydedileceği yer
MAX_INPUT_LENGTH = 512 # Modele verilecek maksimum token sayısı (soru)
MAX_TARGET_LENGTH = 128 # Modelin üreteceği maksimum token sayısı (cevap)
BATCH_SIZE = 4 # GPU belleğine göre ayarlayın (örn: 4, 8, 16)
NUM_EPOCHS = 3 # Eğitim döngüsü sayısı (veri setinize göre ayarlayın)
LEARNING_RATE = 5e-5
PREFIX = "cevapla: " # T5 modeline görevin ne olduğunu belirtmek için ön ek

# --- GPU Kontrolü ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU kullanılabilir: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU bulunamadı, CPU üzerinde çalışılacak.")

# --- Veriyi Yükle ---
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    # Eksik veri varsa temizle (isteğe bağlı ama önerilir)
    df.dropna(subset=['soru', 'cevap'], inplace=True)
    print(f"Veri seti yüklendi. Toplam {len(df)} satır.")
except FileNotFoundError:
    print(f"HATA: '{CSV_PATH}' dosyası bulunamadı!")
    exit()
except Exception as e:
    print(f"HATA: Veri seti yüklenirken bir sorun oluştu: {e}")
    exit()

# Pandas DataFrame'i Hugging Face Dataset objesine çevir
dataset = Dataset.from_pandas(df)

# --- Tokenizer ve Model Yükle ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # Modeli GPU'ya taşı (eğer varsa)
    # model.to(device) # Trainer bunu otomatik yapacak
except Exception as e:
    print(f"HATA: Model veya Tokenizer yüklenirken bir sorun oluştu: {e}")
    exit()

# --- Veri Ön İşleme Fonksiyonu ---
def preprocess_function(examples):
    inputs = [PREFIX + soru for soru in examples['soru']]
    targets = [cevap for cevap in examples['cevap']]

    # Tokenizer'a hem girdileri hem de hedefleri ver
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    # Hedefler (etiketler) için ayrı tokenizasyon
    # padding=True yerine max_length vermek daha güvenli olabilir
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    # Etiketlerdeki padding token'larını -100 ile değiştirerek loss hesaplamasında ignore edilmesini sağla
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Veri Setini İşle ---
print("Veri seti işleniyor...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)
print("Veri seti işlendi.")

# --- Data Collator ---
# Batch'leri dinamik olarak padding yapmak için kullanılır
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100, # Önemli: padding token'larını loss'tan çıkar
    pad_to_multiple_of=8 if torch.cuda.is_available() else None # GPU için 8'in katlarına padding (performans)
)

# --- Eğitim Argümanları ---
# Seq2Seq için özel argümanlar kullanılır
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,          # Model çıktılarının ve checkpoint'lerin kaydedileceği yer
    num_train_epochs=NUM_EPOCHS,              # Toplam eğitim epoch sayısı
    per_device_train_batch_size=BATCH_SIZE,   # Her GPU/CPU için batch boyutu
    # per_device_eval_batch_size=BATCH_SIZE,  # Değerlendirme için batch boyutu (eğer eval seti varsa)
    learning_rate=LEARNING_RATE,              # Öğrenme oranı
    weight_decay=0.01,                        # Ağırlık azaltma (regularizasyon)
    save_total_limit=2,                       # Kaydedilecek toplam checkpoint sayısı
    predict_with_generate=True,               # Değerlendirme/tahmin sırasında generate() kullan
    fp16=torch.cuda.is_available(),           # Eğer GPU varsa Mixed Precision Training (daha hızlı, daha az bellek)
    logging_dir='./logs',                     # TensorBoard loglarının kaydedileceği yer
    logging_steps=100,                        # Her 100 adımda bir loglama yap
    save_strategy="epoch",                    # Her epoch sonunda modeli kaydet
    # evaluation_strategy="epoch",            # Her epoch sonunda değerlendirme yap (eğer eval seti varsa)
    # load_best_model_at_end=True,            # Eğitim sonunda en iyi modeli yükle (eval seti gerektirir)
    # metric_for_best_model="loss",           # En iyi modeli belirlemek için metrik (eval seti gerektirir)
    push_to_hub=False,                        # Modeli Hugging Face Hub'a gönderme
    report_to="none",                         # Logları TensorBoard'a gönder
)

# --- Trainer'ı Oluştur ---
trainer = Seq2SeqTrainer(
    model=model,                         # Eğitilecek model
    args=training_args,                  # Eğitim argümanları
    train_dataset=tokenized_dataset,     # Eğitim veri seti
    # eval_dataset=tokenized_eval_dataset, # Değerlendirme veri seti (varsa)
    tokenizer=tokenizer,                 # Tokenizer
    data_collator=data_collator,         # Data collator
)

# --- Eğitimi Başlat ---
print("Eğitim başlıyor...")
try:
    trainer.train()
    print("Eğitim tamamlandı.")
except Exception as e:
    print(f"HATA: Eğitim sırasında bir sorun oluştu: {e}")
    exit()

# --- Modeli ve Tokenizer'ı Kaydet ---
print(f"Model ve tokenizer '{OUTPUT_DIR}' dizinine kaydediliyor...")
trainer.save_model() # Modeli kaydeder (pytorch_model.bin, config.json vb.)
tokenizer.save_pretrained(OUTPUT_DIR) # Tokenizer'ı kaydeder (tokenizer_config.json, spiece.model vb.)
print("Kayıt işlemi tamamlandı.")

print(f"\nEğitim süreci bitti. Eğitilmiş modeliniz '{OUTPUT_DIR}' klasöründe.")
print("Şimdi 'app.py' dosyasını çalıştırarak modeli test edebilirsiniz.")