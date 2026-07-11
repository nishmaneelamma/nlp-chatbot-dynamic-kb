import streamlit as st
import pandas as pd
import os
from difflib import SequenceMatcher

try:
    import google.generativeai as genai
    gemini_ok = True
except:
    gemini_ok = False

st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.chat-user {
    background: #2563eb;
    color: white;
    padding: 10px 14px;
    border-radius: 14px 14px 4px 14px;
    margin: 6px 0;
    max-width: 70%;
    margin-left: auto;
    font-size: 14px;
}
.chat-bot {
    background: white;
    color: #1e293b;
    padding: 10px 14px;
    border-radius: 14px 14px 14px 4px;
    margin: 6px 0;
    max-width: 80%;
    border: 1px solid #e2e8f0;
    font-size: 14px;
}
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# load the medquad dataset
@st.cache_data
def load_data():
    df = pd.read_csv("medquad.csv")
    df = df.dropna(subset=["answer"])
    df["question_lower"] = df["question"].str.lower()
    df["focus_area"] = df["focus_area"].fillna("General")
    print("dataset loaded, rows:", len(df))
    return df


# keywords for medical entity detection
# i made these lists by going through the dataset manually
DISEASES = ["cancer", "diabetes", "stroke", "alzheimer", "parkinson", "asthma",
            "arthritis", "glaucoma", "leukemia", "pneumonia", "hepatitis", "hiv",
            "aids", "tuberculosis", "epilepsy", "hypertension", "anemia",
            "depression", "anxiety", "dementia", "osteoporosis", "cholesterol"]

SYMPTOMS = ["pain", "fever", "cough", "fatigue", "nausea", "vomiting", "headache",
            "dizziness", "swelling", "rash", "bleeding", "shortness of breath",
            "chest pain", "weight loss", "insomnia", "numbness", "weakness",
            "blurred vision", "diarrhea", "constipation", "itching"]

TREATMENTS = ["treatment", "therapy", "surgery", "medication", "chemotherapy",
              "radiation", "dialysis", "transplant", "vaccine", "physiotherapy",
              "diet", "exercise", "supplement"]

DRUGS = ["aspirin", "insulin", "metformin", "ibuprofen", "paracetamol",
         "antibiotic", "antiviral", "steroid", "antidepressant", "penicillin"]


def find_entities(text):
    # check which medical terms appear in the text
    t = text.lower()
    found = {"diseases": [], "symptoms": [], "treatments": [], "drugs": []}
    for kw in DISEASES:
        if kw in t:
            found["diseases"].append(kw.title())
    for kw in SYMPTOMS:
        if kw in t:
            found["symptoms"].append(kw.title())
    for kw in TREATMENTS:
        if kw in t:
            found["treatments"].append(kw.title())
    for kw in DRUGS:
        if kw in t:
            found["drugs"].append(kw.title())
    return found


def show_entity_tags(entities):
    html = ""
    for e in entities["diseases"]:
        html += f"<span class='tag' style='background:#fee2e2;color:#991b1b'>Disease: {e}</span> "
    for e in entities["symptoms"]:
        html += f"<span class='tag' style='background:#fef3c7;color:#92400e'>Symptom: {e}</span> "
    for e in entities["treatments"]:
        html += f"<span class='tag' style='background:#d1fae5;color:#065f46'>Treatment: {e}</span> "
    for e in entities["drugs"]:
        html += f"<span class='tag' style='background:#ede9fe;color:#5b21b6'>Drug: {e}</span> "
    return html


def get_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def search_dataset(query, df, top_k=3):
    # search the medquad dataset for relevant answers
    # uses keyword matching + similarity score
    q = query.lower()
    scores = []

    for idx, row in df.iterrows():
        score = 0.0

        # word overlap
        q_words = set(q.split())
        row_words = set(row["question_lower"].split())
        common = q_words & row_words
        score += len(common) * 0.4

        # focus area match
        if row["focus_area"].lower() in q:
            score += 1.5

        # similarity
        score += get_similarity(query, row["question"]) * 2.0

        scores.append((score, idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    results = [df.loc[idx] for score, idx in scores[:top_k] if score > 0.3]
    return results


def ask_gemini(query, api_key):
    # fallback to gemini when no good match found in dataset
    if not gemini_ok or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a medical information assistant. "
            "Answer this medical question clearly. "
            "Always remind the user to consult a doctor.\n\n"
            f"Question: {query}"
        )
        r = model.generate_content(prompt)
        return r.text
    except Exception as e:
        print("gemini error:", e)
        return None


# session state
if "chat" not in st.session_state:
    st.session_state.chat = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.title("Medical Q&A Chatbot")
st.caption("Using the MedQuAD dataset — medical questions from NIH and CDC")

df = load_data()

left, right = st.columns([2, 1])

with left:
    chat_box = st.container(height=460)
    with chat_box:
        if not st.session_state.chat:
            st.markdown("<p style='color:#94a3b8;text-align:center;padding:60px 0'>Ask a medical question to get started.</p>",
                        unsafe_allow_html=True)
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-user'>You: {msg['content']}</div>",
                            unsafe_allow_html=True)
            else:
                entity_html = show_entity_tags(find_entities(msg["content"]))
                st.markdown(
                    f"<div class='chat-bot'><b>Bot:</b> {msg['content']}"
                    f"<br><br>{entity_html}"
                    f"<br><small style='color:#94a3b8'>Source: {msg.get('source','MedQuAD')}</small>"
                    f"</div>",
                    unsafe_allow_html=True)

    c1, c2 = st.columns([4, 1])
    with c1:
        user_input = st.text_input("Your question", placeholder="e.g. What are the symptoms of diabetes?",
                                   label_visibility="collapsed")
    with c2:
        ask = st.button("Ask", use_container_width=True)

    if ask and user_input.strip():
        st.session_state.chat.append({"role": "user", "content": user_input})

        results = search_dataset(user_input, df)

        if results:
            top = results[0]
            answer = top["answer"]
            source = f"MedQuAD / {top['source']} ({top['focus_area']})"
            st.session_state.chat.append({
                "role": "bot", "content": answer, "source": source
            })
        else:
            # try gemini if nothing found in dataset
            answer = ask_gemini(user_input, st.session_state.api_key)
            if answer:
                st.session_state.chat.append({
                    "role": "bot", "content": answer, "source": "Gemini AI"
                })
            else:
                st.session_state.chat.append({
                    "role": "bot",
                    "content": "Sorry, I could not find an answer. Please consult a doctor.",
                    "source": "–"
                })
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.rerun()

with right:
    st.markdown("### Dataset Info")
    st.metric("Total Q&A Pairs", f"{len(df):,}")
    st.metric("Topics", df["focus_area"].nunique())

    st.markdown("---")
    st.markdown("**Entity Types**")
    st.markdown("""
    <span class='tag' style='background:#fee2e2;color:#991b1b'>Disease</span>
    <span class='tag' style='background:#fef3c7;color:#92400e'>Symptom</span>
    <span class='tag' style='background:#d1fae5;color:#065f46'>Treatment</span>
    <span class='tag' style='background:#ede9fe;color:#5b21b6'>Drug</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Browse Topics**")
    topics = sorted(df["focus_area"].unique().tolist())
    selected = st.selectbox("Pick a topic", ["All"] + topics)
    if selected != "All":
        filtered = df[df["focus_area"] == selected][["question"]].head(5)
        for _, row in filtered.iterrows():
            st.markdown(f"- {row['question']}")

    st.markdown("---")
    st.markdown("**Gemini API Key (optional)**")
    st.caption("Used when no dataset match is found")
    key_input = st.text_input("API Key", type="password",
                               value=st.session_state.api_key,
                               label_visibility="collapsed")
    if key_input:
        st.session_state.api_key = key_input

    st.markdown("---")
    st.caption("For informational purposes only. Always consult a doctor for personal medical advice.")
