from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("tinyllama-qa-lora", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("tinyllama-qa-lora")

soru = "Einsatzgruppen'in kuruluşu ve işlevi nedir?"
input_text = f"Soru: {soru}\nCevap:"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
"""
# v1
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
"""

# v2
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    num_beams=4,
    no_repeat_ngram_size=2,
    early_stopping=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
"""
# v3
outputs = model.generate(
    **inputs,
    max_new_tokens=256,  # 128 yerine 256, Türkçe uzun cevaplar için yeterli
    num_beams=4,  # Beam search ile daha kaliteli cevaplar
    no_repeat_ngram_size=2,  # Aynı ngramların tekrarını engelle
    early_stopping=True,  # Cevap bitmeden durma
    temperature=0.7,  # Daha tutarlı ve anlamlı cevaplar
    top_p=0.9,
    do_sample=True,  # Farklı varyasyonlar için sampling kullan
    pad_token_id=tokenizer.eos_token_id,  # Padding token'ı bitiş token'ı ile değiştir
    eos_token_id=tokenizer.eos_token_id  # Cevap bittiğinde modelin durması için
)
"""
cevap = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Cevap: ")[1].strip()
print(soru)
print(cevap)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))