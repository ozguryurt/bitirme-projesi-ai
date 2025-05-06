from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_name = "savasy/bert-base-turkish-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

print(qa_pipeline(question="Flutter nedir?", context="Flutter, Google tarafından geliştirilen açık kaynaklı bir UI (Kullanıcı Arayüzü) geliştirme kitidir (SDK). Tek bir kod tabanı kullanarak mobil (Android, iOS), web, masaüstü (Windows, macOS, Linux) ve gömülü sistemler için görsel olarak çekici, hızlı ve yerel olarak derlenmiş uygulamalar oluşturmak için kullanılır. Kendi yüksek performanslı Skia grafik motorunu kullanarak UI'ı doğrudan çizer."))

"""
Output:
açık kaynaklı bir UI (Kullanıcı Arayüzü) geliştirme kitidir
"""