#pip install PyJWT
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import jwt
import random
from functools import wraps
from datetime import datetime

# 🔹 QA ve embedding modelleri
#qa_pipeline = pipeline("question-answering", model="savasy/bert-base-turkish-squad")
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="./models/qa_model")
embedding_model = SentenceTransformer('./models/embedding_model')

# 🔹 Örnek veritabanı
questions = [
    {
        "id": 1,
        "question_text": "Flutter nedir?"
    },
    {
        "id": 2,
        "question_text": "State management ne işe yarar?"
    },
    {
        "id": 3,
        "question_text": "Dart dili ile Flutter arasındaki fark nedir?"
    },
    {
        "id": 4,
        "question_text": "Flutter'da widget nasıl çalışır?"
    },
    {
        "id": 5,
        "question_text": "Flutter'ın avantajları nelerdir?"
    },
    {
        "id": 6,
        "question_text": "Hot Reload ve Hot Restart arasındaki fark nedir?"
    },
    {
        "id": 7,
        "question_text": "Flutter'da layout (yerleşim) nasıl yapılır?"
    },
    {
        "id": 8,
        "question_text": "Popüler Flutter state management çözümleri nelerdir?"
    },
    {
        "id": 9,
        "question_text": "Flutter neden yüksek performanslıdır?"
    },
    {
        "id": 10,
        "question_text": "Flutter projesinin temel yapısı nasıldır?"
    }
]

answers = [
    {
        "id": 1,
        "question_id": 1,
        "answer_text": "Flutter, Google tarafından geliştirilen açık kaynaklı bir UI (Kullanıcı Arayüzü) geliştirme kitidir (SDK)."
    },
    {
        "id": 2,
        "question_id": 1,
        "answer_text": "Tek bir kod tabanı kullanarak mobil (Android, iOS), web, masaüstü (Windows, macOS, Linux) ve gömülü sistemler için görsel olarak çekici, hızlı ve yerel olarak derlenmiş uygulamalar oluşturmak için kullanılır."
    },
    {
        "id": 3,
        "question_id": 1,
        "answer_text": "Kendi yüksek performanslı Skia grafik motorunu kullanarak UI'ı doğrudan çizer."
    },
    {
        "id": 4,
        "question_id": 2,
        "answer_text": "State management, bir Flutter uygulamasındaki verilerin (state) durumunu yönetme ve bu durum değiştiğinde kullanıcı arayüzünün (UI) otomatik olarak güncellenmesini sağlama işlemidir."
    },
    {
        "id": 5,
        "question_id": 2,
        "answer_text": "Uygulama büyüdükçe veri akışını kontrol altında tutmayı, bileşenler arası veri paylaşımını kolaylaştırmayı ve kodun daha organize olmasını sağlar."
    },
    {
        "id": 6,
        "question_id": 3,
        "answer_text": "Dart, Flutter uygulamalarını yazmak için kullanılan nesne yönelimli bir programlama dilidir. Google tarafından geliştirilmiştir."
    },
    {
        "id": 7,
        "question_id": 3,
        "answer_text": "Flutter ise Dart dilini kullanan bir UI framework'ü ve SDK'sıdır. Uygulama arayüzünü oluşturmak için gerekli araçları, widget'ları ve kütüphaneleri sağlar."
    },
    {
        "id": 8,
        "question_id": 3,
        "answer_text": "Kısacası, Dart 'dil', Flutter ise o dili kullanarak uygulama geliştirmeyi sağlayan 'araç seti'dir."
    },
    {
        "id": 9,
        "question_id": 4,
        "answer_text": "Widget'lar, Flutter'da kullanıcı arayüzünü oluşturan temel yapı taşlarıdır. Ekrandaki her şey (butonlar, metinler, resimler, layout'lar vb.) bir widget'tır."
    },
    {
        "id": 10,
        "question_id": 4,
        "answer_text": "Flutter, widget'ları bir ağaç yapısında (widget tree) birleştirerek arayüzü oluşturur. Her widget, kendi 'build' metodunda nasıl görüneceğini ve davranacağını tanımlar."
    },
    {
        "id": 11,
        "question_id": 4,
        "answer_text": "Temelde iki tür widget vardır: StatelessWidget (durumu değişmeyen) ve StatefulWidget (durumu değişebilen ve bu değişikliklere göre yeniden çizilebilen)."
    },
    {
        "id": 12,
        "question_id": 5,
        "answer_text": "Hızlı Geliştirme: Hot Reload özelliği sayesinde kod değişiklikleri anında görülür."
    },
    {
        "id": 13,
        "question_id": 5,
        "answer_text": "Etkileyici ve Esnek UI: Zengin widget kütüphanesi ve özelleştirme imkanları sunar."
    },
    {
        "id": 14,
        "question_id": 5,
        "answer_text": "Yüksek Performans: Kod doğrudan yerel ARM makine koduna derlendiği için yüksek performans sunar."
    },
    {
        "id": 15,
        "question_id": 5,
        "answer_text": "Tek Kod Tabanı: Farklı platformlar için tek bir kod yazarak geliştirme süresini ve maliyetini azaltır."
    },
    {
        "id": 16,
        "question_id": 6,
        "answer_text": "Hot Reload: Kod değişikliklerini çalışan uygulamaya enjekte eder, uygulamanın mevcut durumunu (state) korur. Genellikle saniyeler sürer. UI değişiklikleri için idealdir."
    },
    {
        "id": 17,
        "question_id": 6,
        "answer_text": "Hot Restart: Uygulamanın mevcut durumunu sıfırlar ve uygulamayı yeniden başlatır. Hot Reload'dan daha yavaştır ama state değişiklikleri veya bazı köklü değişiklikler gerektiğinde kullanılır."
    },
    {
        "id": 18,
        "question_id": 7,
        "answer_text": "Flutter'da layout, widget'lar kullanılarak yapılır. `Row` (yatay), `Column` (dikey), `Stack` (üst üste), `Container`, `Padding`, `Center`, `Align`, `Expanded` gibi widget'lar bileşenleri ekranda konumlandırmak ve düzenlemek için kullanılır."
    },
    {
        "id": 19,
        "question_id": 7,
        "answer_text": "Widget'lar iç içe geçirilerek karmaşık layout yapıları oluşturulabilir."
    },
    {
        "id": 20,
        "question_id": 8,
        "answer_text": "Provider: Basit ve yaygın kullanılan, InheritedWidget üzerine kurulu bir çözümdür."
    },
    {
        "id": 21,
        "question_id": 8,
        "answer_text": "Riverpod: Provider'ın geliştiricisi tarafından oluşturulan, daha esnek ve derleme zamanı güvenliği sunan modern bir çözümdür."
    },
    {
        "id": 22,
        "question_id": 8,
        "answer_text": "Bloc/Cubit: Özellikle büyük ve karmaşık uygulamalarda state yönetimini ve iş mantığını ayırmak için kullanılan bir desendir."
    },
    {
        "id": 23,
        "question_id": 8,
        "answer_text": "GetX: State management, dependency injection ve route management gibi birçok özelliği bir arada sunan bir mikro framework'tür."
    },
    {
        "id": 24,
        "question_id": 9,
        "answer_text": "Flutter, UI'ı çizmek için platformun yerel UI bileşenlerini (OEM widgets) kullanmak yerine kendi Skia grafik motorunu kullanır. Bu, platform farklılıklarından kaynaklanan performans sorunlarını ortadan kaldırır."
    },
    {
        "id": 25,
        "question_id": 9,
        "answer_text": "Release modunda Dart kodu doğrudan yerel ARM veya x64 makine koduna (Ahead-of-Time - AOT compilation) derlenir, bu da JavaScript köprüsü gibi ara katmanlara ihtiyaç duymadan yüksek hızda çalışmasını sağlar."
    },
    {
        "id": 26,
        "question_id": 10,
        "answer_text": "Bir Flutter projesi genellikle şu ana dizinleri içerir: `lib` (Dart kodunun bulunduğu ana dizin, `main.dart` başlangıç dosyasıdır), `android` / `ios` (platforma özgü proje dosyaları), `web`, `windows`, `linux`, `macos` (diğer platformlar için dosyalar), `test` (test kodları), `pubspec.yaml` (proje bağımlılıklarını ve meta verilerini tanımlayan dosya)."
    }
]

def get_answers_by_question_id(q_id):
    return [a["answer_text"] for a in answers if a["question_id"] == q_id]

def find_top_matching_questions(user_question, top_n=3):
    question_texts = [q["question_text"] for q in questions]
    embeddings = embedding_model.encode(question_texts, convert_to_tensor=True)
    user_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
    similarity_scores = util.cos_sim(user_embedding, embeddings)[0]

    top_indices = torch.topk(similarity_scores, top_n).indices.tolist()
    top_questions = []
    for idx in top_indices:
        q_id = questions[idx]["id"]
        score = similarity_scores[idx].item()
        top_questions.append({
            "id": q_id,
            "question_text": questions[idx]["question_text"],
            "similarity_score": score
        })

    return top_questions

def generate_comprehensive_answer(user_question, top_matched_questions):
    all_content = ""
    used_answers = set()

    for q in top_matched_questions:
        q_id = q["id"]
        all_answers = get_answers_by_question_id(q_id)
        for answer in all_answers:
            if answer not in used_answers:
                all_content += answer + " "
                used_answers.add(answer)

    result = qa_pipeline(question=user_question, context=all_content)
    comprehensive_answer = f"{result['answer']}\n\nDaha detaylı bilgi:\n\n"

    for q in top_matched_questions:
        q_id = q["id"]
        q_text = q["question_text"]
        q_answers = get_answers_by_question_id(q_id)
        comprehensive_answer += f"📌 {q_text}\n"
        for i, ans in enumerate(q_answers, 1):
            comprehensive_answer += f"{i}. {ans}\n"
        comprehensive_answer += "\n"

    return comprehensive_answer

# Flask APP

app = Flask(__name__)
CORS(
    app,
    supports_credentials=True,  # credentials'a izin ver
    resources={
        r"/ask-ai": {
            "origins": "http://localhost:5173",  # Sadece bu origin'e izin ver
            "methods": ["POST", "OPTIONS"],  # İzin verilen metodlar
            "allow_headers": ["Content-Type"],  # İzin verilen header'lar
        }
    }
)
app.config['JWT_SECRET_KEY'] = 'SECRET'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Cookie'den token'ı alıyoruz
        if 'token' in request.cookies:
            token = request.cookies.get('token')
        
        if not token:
            return jsonify({'error': 'Yetkisiz erişim.'}), 401
        
        try:
            # Token doğrulama
            jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Yetkisiz erişim.'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Yetkisiz erişim.'}), 401
            
        return f(*args, **kwargs)
    
    return decorated

@app.route("/ask-ai", methods=["POST"])
@token_required
def ask():
    data = request.get_json()
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "Soru eksik."}), 400

    top_matched_questions = find_top_matching_questions(user_question, top_n=3)
    answer = generate_comprehensive_answer(user_question, top_matched_questions)

    return jsonify({
        "user": {
                "id": random.randint(0, 300),
                "content": user_question,
                "role": "user",
                "timestamp": datetime.now().isoformat()
        },
        "assistant": {
                "id": random.randint(0, 300),
                "content": answer,
                "role": "assistant",
                "timestamp": datetime.now().isoformat()
        }
    })

if __name__ == "__main__":
    app.run(debug=True)