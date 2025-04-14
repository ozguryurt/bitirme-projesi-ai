import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Ayarlar ---
MODEL_DIR = "./t5_soru_cevap_model"  # train.py'nin modeli kaydettiği klasör
PREFIX = "cevapla: "                 # Eğitim sırasında kullanılan ön ek ile aynı olmalı!
MAX_INPUT_LENGTH = 512              # Eğitimle aynı olmalı
MAX_GENERATION_LENGTH = 128         # Üretilecek cevabın maksimum uzunluğu (isteğe bağlı)

# --- GPU Kontrolü ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU kullanılabilir: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU bulunamadı, CPU üzerinde çalışılacak.")

# --- Eğitilmiş Modeli ve Tokenizer'ı Yükle ---
print(f"'{MODEL_DIR}' dizininden model ve tokenizer yükleniyor...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval() # Modeli inference moduna al (dropout vb. katmanları kapatır)
    print("Model ve tokenizer başarıyla yüklendi.")
except OSError:
    print(f"HATA: '{MODEL_DIR}' dizininde kaydedilmiş model bulunamadı.")
    print("Lütfen önce 'train.py' betiğini çalıştırarak modeli eğitin ve kaydedin.")
    exit()
except Exception as e:
    print(f"HATA: Model veya tokenizer yüklenirken bir sorun oluştu: {e}")
    exit()

# --- Cevap Üretme Fonksiyonu ---
def generate_answer(question):
    input_text = PREFIX + question

    inputs = tokenizer(input_text,
                       return_tensors="pt",
                       max_length=MAX_INPUT_LENGTH,
                       padding=True,
                       truncation=True)

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_GENERATION_LENGTH,
            num_beams=5,  # Beam search parametresi
            early_stopping=True
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# --- Konsol Uygulaması Döngüsü ---
print("\n--- Türkçe Soru-Cevap Modeli ---")
print("Sormak istediğiniz soruyu girin (Çıkmak için 'çıkış' yazın).")

while True:
    user_input = input("\nSoru: ")
    if user_input.lower() == 'çıkış':
        print("Uygulamadan çıkılıyor...")
        break
    if not user_input:
        continue

    try:
        print("Cevap üretiliyor...")
        answer = generate_answer(user_input)
        print(f"Modelin Cevabı: {answer}")
    except Exception as e:
        print(f"HATA: Cevap üretilirken bir sorun oluştu: {e}")