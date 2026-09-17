import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Multi-LLM Debate & Consensus", layout="wide")
st.title("🤖 Yapay Zeka Çapraz Değerlendirme ve Ortak Karar Arayüzü")

# Yan menüden OpenRouter API anahtarını alalım
api_key = st.sidebar.text_input("OpenRouter API Key Girin:", type="password")

if not api_key:
    st.info("Devam etmek için lütfen yan menüden OpenRouter API anahtarınızı girin.")
    st.stop()

# OpenRouter istemcisi (OpenAI SDK'sı ile uyumludur)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Kullanılacak Modeller
MODEL_A = "openai/gpt-4o"
MODEL_B = "anthropic/claude-3.5-sonnet"
# Hakem / Sentez Modeli
SYNTHESIZER_MODEL = "openai/gpt-4o" 

def get_llm_response(model: str, prompt: str) -> str:
    """Belirtilen modele prompt gönderir ve yanıt döner."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# Kullanıcı Girdisi
user_prompt = st.text_area("Promptunuzu yazın:", height=100)

if st.button("Modelleri Çalıştır, Değerlendir ve Sentezle", type="primary"):
    if user_prompt.strip():
        
        # ==========================================
        # 1. AŞAMA: İLK YANITLAR
        # ==========================================
        st.header("1. Aşama: Bağımsız Yanıtlar")
        with st.spinner("Modeller ilk yanıtlarını üretiyor..."):
            ans_a = get_llm_response(MODEL_A, user_prompt)
            ans_b = get_llm_response(MODEL_B, user_prompt)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🔴 {MODEL_A}")
            st.write(ans_a)
        with col2:
            st.subheader(f"🔵 {MODEL_B}")
            st.write(ans_b)

        st.divider()

        # ==========================================
        # 2. AŞAMA: ÇAPRAZ DEĞERLENDİRME
        # ==========================================
        st.header("2. Aşama: Çapraz Eleştiri ve Puanlama")
        with st.spinner("Modeller birbirlerinin yanıtlarını inceliyor..."):
            eval_prompt_for_a = f"""
Soru: {user_prompt}

Başka bir yapay zekanın bu soruya verdiği yanıt:
---
{ans_b}
---

Lütfen bu yanıtı doğruluk, mantık, eksiklikler ve netlik açısından değerlendir. 10 üzerinden puan ver, güçlü ve zayıf yönlerini belirt.
"""
            eval_prompt_for_b = f"""
Soru: {user_prompt}

Başka bir yapay zekanın bu soruya verdiği yanıt:
---
{ans_a}
---

Lütfen bu yanıtı doğruluk, mantık, eksiklikler ve netlik açısından değerlendir. 10 üzerinden puan ver, güçlü ve zayıf yönlerini belirt.
"""

            eval_a = get_llm_response(MODEL_A, eval_prompt_for_a)
            eval_b = get_llm_response(MODEL_B, eval_prompt_for_b)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader(f"{MODEL_A}'nın Eleştirisi:")
            st.info(eval_a)
        with col4:
            st.subheader(f"{MODEL_B}'nin Eleştirisi:")
            st.warning(eval_b)

        st.divider()

        # ==========================================
        # 3. AŞAMA: ORTAK KONSENSÜS / SENTEZ YANIT
        # ==========================================
        st.header("🎯 3. Aşama: Nihai Ortak Yanıt (Sentez)")
        with st.spinner("Tüm yanıtlar ve eleştiriler harmanlanarak ortak karar oluşturuluyor..."):
            
            synthesis_prompt = f"""
Sen uzman bir hakem ve yapay zeka sentezleyicisisin.

KULLANICININ İLK SORUSU:
{user_prompt}

----------------------------------------
MODEL A ({MODEL_A}) YANITI:
{ans_a}

MODEL B'NİN MODEL A HAKKINDAKİ ELEŞTİRİSİ:
{eval_b}
----------------------------------------
MODEL B ({MODEL_B}) YANITI:
{ans_b}

MODEL A'NIN MODEL B HAKKINDAKİ ELEŞTİRİSİ:
{eval_a}
----------------------------------------

GÖREVİN:
1. İki modelin doğru, güçlü ve faydalı noktalarını birleştir.
2. Eleştirilerde tespit edilen hata, eksiklik veya yanlış yönlendirmeleri ayıkla.
3. Kullanıcıya, her iki modelin de üzerinde uzlaşacağı, en doğru, eksiksiz, tutarlı ve net "NİHAİ ORTAK YANITI" sun.
"""
            consensus_answer = get_llm_response(SYNTHESIZER_MODEL, synthesis_prompt)

        # Ortak yanıtı belirgin bir kutu içerisinde gösterelim
        st.success("### 🏆 Konsensüs Yanıtı")
        st.write(consensus_answer)
