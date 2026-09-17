import streamlit as st
import os
from openai import OpenAI

# 1. SAYFA AYARLARI
icon_path = os.path.join(os.path.dirname(__file__), 'app_icon.png')
page_icon = icon_path if os.path.exists(icon_path) else "🧠"

st.set_page_config(
    page_title="Yapay Zeka Kolektivizmi",
    page_icon=page_icon,
    layout="wide"
)

st.title("🧠 Yapay Zeka Kolektivizmi (Multi-Agent Consensus)")
st.caption("Farklı yapay zeka modelleri tartışır, ortak paydayı bulur.")

# 2. YAN MENÜ (API KEY VE MODEL SEÇİMİ)
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("OpenRouter API Key", type="password", help="sk-or-v1-... ile başlayan anahtarınızı girin")
    
    st.divider()
    st.subheader("Ücretsiz Modeller")
    
    # Tüm modelleri varsayılan olarak %100 ücretsiz olanlardan seçtik
    agent1_model = st.selectbox(
        "1. Ajan Modeli",
        ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-ultra:free", "openai/gpt-oss-20b:free", "openrouter/free"]
    )
    
    agent2_model = st.selectbox(
        "2. Ajan Modeli",
        ["nvidia/nemotron-3-ultra:free", "google/gemma-4-31b-it:free", "openai/gpt-oss-20b:free", "openrouter/free"]
    )
    
    judge_model = st.selectbox(
        "Konsensüs / Sentez Modeli",
        ["openrouter/free", "google/gemma-4-31b-it:free", "nvidia/nemotron-3-ultra:free"]
    )

# 3. ANA UYGULAMA MANTIĞI
prompt = st.text_area("Yapay zeka kuruluna sorunuzu yazın:", height=120, placeholder="Örn: Yapay zekanın tıp eğitimindeki geleceği ne olacak?")

if st.button("Kolektif Akla Sor", type="primary"):
    if not api_key:
        st.error("Lütfen sol yan menüden OpenRouter API Key anahtarınızı girin!")
    elif not prompt.strip():
        st.warning("Lütfen bir soru yazın.")
    else:
        # OpenRouter istemcisini ilklendir
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        try:
            # 1. Aşama: Ajanların Görüş Bildirmesi
            st.subheader("💬 1. Aşama: Bağımsız Ajan Görüşleri")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Ajan 1 (`{agent1_model}`):**")
                with st.spinner("Ajan 1 yanıt üretiyor..."):
                    res1 = client.chat.completions.create(
                        model=agent1_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ans1 = res1.choices[0].message.content
                    st.info(ans1)

            with col2:
                st.markdown(f"**Ajan 2 (`{agent2_model}`):**")
                with st.spinner("Ajan 2 yanıt üretiyor..."):
                    res2 = client.chat.completions.create(
                        model=agent2_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ans2 = res2.choices[0].message.content
                    st.success(ans2)

            # 2. Aşama: Sentez ve Konsensüs
            st.divider()
            st.subheader("🎯 2. Aşama: Kolektif Konsensüs & Sentez Raporu")
            
            synthesis_prompt = f"""
            Aşağıda bir kullanıcı sorusu ve iki farklı yapay zeka ajanının bu soruya verdiği yanıtlar verilmiştir.
            
            Kullanıcı Sorusu: {prompt}
            
            Ajan 1 Yanıtı:
            {ans1}
            
            Ajan 2 Yanıtı:
            {ans2}
            
            Görevin:
            1. İki yanıt arasındaki ortak noktaları ve çelişkileri analiz et.
            2. Her iki tarafın da en doğru fikirlerini birleştirerek tek bir nihai 'Kolektif Konsensüs Raporu' oluştur.
            """

            with st.spinner(f"Konsensüs Modeli (`{judge_model}`) sentezliyor..."):
                res_judge = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": synthesis_prompt}]
                )
                final_consensus = res_judge.choices[0].message.content
                st.markdown(final_consensus)

        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
