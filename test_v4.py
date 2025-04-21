from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch

# Türkçe BERT modelini ve tokenizer'ı yükle
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Soru-cevap için modeli yükle
# Not: İlk başta genel BERT modelini yükleyip sonra fine-tune etmeniz gerekecek
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Soru-cevap pipeline'ı oluştur
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Örnek kullanım
def answer_question(question, context):
    result = qa_pipeline({
        'question': question,
        'context': context
    })
    return result

# Test
context = "1881 yılında Selanik'te doğmuştur."
question = "Atatürk nerede doğmuştur?"

answer = answer_question(question, context)
print(f"Soru: {question}")
print(f"Cevap: {answer['answer']}")
print(f"Güven skoru: {answer['score']:.4f}")