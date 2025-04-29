from transformers import AutoTokenizer, pipeline, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

# QA modeli ve tokenizer'ı indir ve kaydet
model_name = "savasy/bert-base-turkish-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Pipeline oluştur
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Model ve tokenizer'ı kaydet
model.save_pretrained("./models/qa_model")
tokenizer.save_pretrained("./models/qa_model")

# SentenceTransformer modelini indir ve kaydet
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_model.save('./models/embedding_model')
