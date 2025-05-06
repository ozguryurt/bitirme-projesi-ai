from transformers import pipeline, GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

soru = "Soru: Flutter nedir?\nCevap:"
sonuc = generator(
    soru, 
    max_length=150,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,  # Daha düşük sıcaklık daha tutarlı çıktılar verir
    top_p=0.9,        # Nucleus sampling
    top_k=50,         # En olası 50 tokeni dikkate al
    repetition_penalty=1.2,  # Tekrarları azaltmak için ceza
    no_repeat_ngram_size=2   # 2-gram tekrarlarını engelle
)
print(sonuc[0]["generated_text"])

"""
Output:
Soru: Flutter nedir?
Cevap: Tek Kod Tabanı, yerel olarak derlenmiş uygulamalar oluşturmak için kullanılır. Uygulama geliştirme süreci hızlı ve kolaydır. Birçok kod tabanı mevcut olduğundan (state management), kullanıcılar kendi uygulamalarını oluşturabilirler. Her platform için bir UI widget'ı mevcuttur.
Summary: State manage ne işe yarar? (State Management)
Dart dili, state yönetimi için birçok özellik sağlar. Skia: Widgets in the UX framework
Build: application integration
Expanded: container states and manages
OEM: developers interface
Vienna: stack manegement
R
"""