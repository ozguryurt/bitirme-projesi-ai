import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import time

model_name = "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
    load_in_8bit_fp32_cpu_offload=True,
    device_map = 'auto'
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    temperature=0.3,
    repetition_penalty=1.1,
    top_p=0.9,
    max_length=610,
    do_sample=True,
    return_full_text=False,
    min_new_tokens=32
)

text = """Yapay zeka hakkında 3 tespit yaz.\n"""

r = text_generator(text)

print(r[0]['generated_text'])

"""
1. Yapay Zeka (AI), makinelerin insan benzeri bilişsel işlevleri gerçekleştirmesini sağlayan bir teknoloji alanıdır.

2. Yapay zekanın geliştirilmesi ve uygulanması, sağlık hizmetlerinden eğlenceye kadar çeşitli sektörlerde çok sayıda fırsat sunmaktadır.

3. Yapay zeka teknolojisinin potansiyel faydaları önemli olsa da mahremiyet, işten çıkarma ve etik hususlar gibi konularla ilgili endişeler de var.
"""