import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class TurkceChatbot:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print("Model yükleniyor, lütfen bekleyin...")

        # Model ve tokenizer'ı yükle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPU varsa kullan, yoksa CPU kullan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cihaz: {self.device}")

        # Modeli belirtilen cihaza yükle
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        # Text generation pipeline oluştur
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        print("Model başarıyla yüklendi!")

    def generate_response(self, prompt):
        # TinyLlama için Türkçe yanıt vermesini sağlayan özel prompt
        system_prompt = "Sen yardımcı bir yapay zeka asistanısın. Tüm sorulara SADECE TÜRKÇE olarak cevap ver. Cevapların kısa ve öz olsun."

        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"

        # Yanıt üret
        response = self.generator(
            formatted_prompt,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Yanıtı formatla ve döndür
        response_text = response[0]['generated_text']

        # Asistan yanıtını ayıkla
        try:
            assistant_response = response_text.split("<|assistant|>\n")[-1].strip()
        except:
            # Eğer splitleme çalışmazsa, tüm yanıtı döndür
            assistant_response = response_text

        return assistant_response

    def clean_response(self, text):
        """Yanıttaki olası format etiketlerini temizler"""
        # Olası EOS token ve diğer kontrol karakterlerini temizle
        cleanup_tokens = ["</s>", "<|endoftext|>", "<|user|>", "<|system|>"]
        for token in cleanup_tokens:
            text = text.replace(token, "")

        # Yanıtın sonunda başka bir konuşma başlangıcı varsa kes
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[0]

        return text.strip()

    def run(self):
        print("Türkçe Chatbot başlatıldı. Çıkmak için 'çıkış' yazın.")

        while True:
            user_input = input("\nSiz: ")

            if user_input.lower() in ["çıkış", "kapat", "quit", "exit"]:
                print("Chatbot kapatılıyor...")
                break

            # Türkçe yanıt vermesi için ek yönerge
            if not any(word in user_input.lower() for word in ["türkçe", "türkçe yanıt", "türkçe cevap"]):
                user_input = user_input + " (Lütfen Türkçe cevap ver.)"

            response = self.generate_response(user_input)
            cleaned_response = self.clean_response(response)
            print(f"\nChatbot: {cleaned_response}")


if __name__ == "__main__":
    # TinyLlama modeli kullan
    chatbot = TurkceChatbot(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # chatbot = TurkceChatbot(model_name="tiiuae/falcon-7b")
    # chatbot = TurkceChatbot(model_name="bigscience/bloom-1b7")
    chatbot.run()