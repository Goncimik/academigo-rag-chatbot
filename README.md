---
title: Akademigo RAG Chatbot
emoji: 🎓🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Akademik RAG Chatbot 

RAG tabanlı tez & makale QA chatbot'u. 


# Projenin Amacı

“Bu proje, PDF/TXT tabanlı akademik içerikten hızlı ve kaynaklı yanıt üreten bir RAG (Retrieval-Augmented Generation) chatbot’u sunar. 
Amaç, tez/makale gibi uzun metinlerde doğru pasajları geri çağırıp LLM’e bağlam vererek kaynak sayfa atıflı yanıtlar üretmektir.”

# Veri Seti

“Hugging Face’ten ccdv/pubmed-summarization kullanıldı.

## Özellikler
- 📄 PDF/TXT yükleme
- 🔍 Chunking + Embedding: `all-MiniLM-L6-v2`
- 🧠 Vektör arama: FAISS
- 🤖 Üretici model: `HuggingFaceH4/zephyr-7b-beta`
- 🌐 Gradio arayüz (Index Kur / Soru Sor)

## Kurulum (Spaces)
1. New Space → SDK: *Gradio*, Visibility: *Public*.
2. `app.py`, `requirements.txt`, `README.md` 
3. Secrets kısmı`HF_TOKEN` 
4. Variables kısmı `HF_MODEL` 

## Kullanım
1. Index Kur sekmesinde PDF/TXT dosyalarınızı yükleyin.
2. Soru Sor sekmesinde akademik sorunuzu yazın.

## 💬 Chatbot Geliştirirken Karşılaştığım Sorunlar

🧩 Kaggle’da paketler birbiriyle çakıştı.
⚙️ Colab’da da kod savaşları devam etti.
🤖 Hugging Face’e geçince sistem sakinleşti ama ben token ve API isimlerini karıştırdım.
💡 Sonuç: chatbotum oluştu ama ceavapsız sorular bıraktı.
🚀 Tavsiye: Hugging Face hızlı ve kolay — ama token’ınızı sakın karıştırmayın ve de unutmayın!
🧩 Not:  Chatbotum oluştu ama cevapsız sorular ile. Hatamı düzeltemedim.


# Deploy Linki
##  Academigo RAG Chatbot
[Akademigo RAG Chatbot (Hugging Face Space)](https://huggingface.co/spaces/goncimik/academigo)
