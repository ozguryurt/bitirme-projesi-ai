from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Huggingface modelini diske indir
qa_pipeline = pipeline("question-answering", model="savasy/bert-base-turkish-squad")
qa_pipeline.save_pretrained("./models/qa_model")

# SentenceTransformer modelini indir ve kaydet
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_model.save('./models/embedding_model')