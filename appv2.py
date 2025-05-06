from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("ozcangundes/mt5-small-turkish-squad")
model = AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/mt5-small-turkish-squad")
def get_answer(question,context):
  source_encoding=tokenizer(
    question,
    context,
    max_length=512,
    padding="max_length",
    truncation="only_second",
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt")
  generated_ids=model.generate(
      input_ids=source_encoding["input_ids"],
      attention_mask=source_encoding["attention_mask"],
      max_length=120)

  preds=[tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id in generated_ids]

  return "".join(preds)

question={
    "context":"""Flutter, Google tarafından geliştirilen açık kaynaklı bir UI (Kullanıcı Arayüzü) geliştirme kitidir (SDK). Tek bir kod tabanı kullanarak mobil (Android, iOS), web, masaüstü (Windows, macOS, Linux) ve gömülü sistemler için görsel olarak çekici, hızlı ve yerel olarak derlenmiş uygulamalar oluşturmak için kullanılır. Kendi yüksek performanslı Skia grafik motorunu kullanarak UI'ı doğrudan çizer.""",
    "question":"Flutter nedir?"
    }
    
print(get_answer(question["question"],question["context"]))
"""
Output:
Kendi yüksek performanslı Skia grafik motorunu kullanarak mobil (Android, iOS), web, masaüstü (Windows, MacOS, Linux) ve gömülü sistemler için görsel olarak çekici, hızlı ve yerel olarak derlenmiş uygulamalar oluşturmak için kullanılır.
"""