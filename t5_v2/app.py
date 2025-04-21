# app.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Model yolu güncellendi
model_path = "./quora_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def answer_question(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Konsol arayüzü
if __name__ == "__main__":
    print("Ask a question (type 'exit' to quit):")
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = answer_question(user_input)
        print("\nAnswer:", response)