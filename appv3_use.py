from transformers import pipeline, GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

soru = "Soru: Flutter nedir?\nCevap:"
sonuc = generator(soru, max_length=150)
print(sonuc[0]["generated_text"])

"""
Output:
Soru: Flutter nedir?
Cevap: Flutter, Google tarafından geliştirilen açık kaynaklı bir içerik yönetim sistemidir. Flutter'da yerel olarak derlenen yüksek düzeyde özelleştirme imkanı ve esnekliği olan bir UI seti (OEM widgets) vardır.
Flutter, UI'ı çizmek ve düzenlemek için kullanılan Skia grafik motorunu kullanır. Bu da yüksek hızda çalışmasını, birden fazla platformda çalışabilmesini ve yerel olarak derlendiği için güvenlik açıklarına karşı daha dirençli olmasını sağlar.
Flutter, kendi Skia grafik motorunu kullanarak grafik oluşturmayı, bu grafikleri sunmayı ve geliştirmeyi sağlayan nesne yönelimli bir programlama dilidir.
Flutter, Skia grafik motorunu kullanarak uygulamalar oluşturmak için kullanılan Ski
"""