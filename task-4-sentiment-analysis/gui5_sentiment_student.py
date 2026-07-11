import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import os
import threading

try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
WORDS_PATH = os.path.join(BASE_DIR, "words.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.h5")

THRESHOLD = 0.25

# vader gives compound score from -1 to +1
# above 0.05 = positive, below -0.05 = negative
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.lemmatizer = WordNetLemmatizer()
        self.conversation_history = []
        self.sentiment_log = []
        self.gemini_key = ""

        # setup vader
        nltk.download("vader_lexicon", quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        print("vader loaded ok")

        self.setup_gui()
        self.load_data()

    def setup_gui(self):
        self.master.title("Fitness Chatbot - Sentiment Analysis")
        self.master.geometry("960x680")
        self.master.configure(bg="#0f172a")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0f172a")
        style.configure("TLabel", background="#0f172a", foreground="#e2e8f0")
        style.configure("TButton", background="#1e293b", foreground="#e2e8f0",
                        borderwidth=0, padding=6)
        style.map("TButton",
                  background=[("active", "#3b82f6")],
                  foreground=[("active", "white")])
        style.configure("TEntry", fieldbackground="#1e293b",
                        foreground="#e2e8f0", insertcolor="#e2e8f0")
        style.configure("TNotebook", background="#0f172a")
        style.configure("TNotebook.Tab", background="#1e293b",
                        foreground="#94a3b8", padding=[12, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", "#3b82f6")],
                  foreground=[("selected", "white")])

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        self.api_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.chat_tab,  text="Chat")
        self.notebook.add(self.stats_tab, text="Sentiment Stats")
        self.notebook.add(self.api_tab,   text="API Key")

        self._build_chat_tab()
        self._build_stats_tab()
        self._build_api_tab()

    def _build_chat_tab(self):
        # top bar to show current sentiment
        top = ttk.Frame(self.chat_tab)
        top.pack(fill=tk.X, padx=12, pady=(8, 0))
        ttk.Label(top, text="Detected Sentiment:").pack(side=tk.LEFT)
        self.sentiment_var = tk.StringVar(value="—")
        self.sent_label = tk.Label(top, textvariable=self.sentiment_var,
                                   bg="#0f172a", fg="#94a3b8",
                                   font=("Segoe UI", 10, "bold"))
        self.sent_label.pack(side=tk.LEFT, padx=8)
        self.score_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.score_var,
                  font=("Segoe UI", 9), foreground="#475569").pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Loading...")
        ttk.Label(self.chat_tab, textvariable=self.status_var,
                  font=("Segoe UI", 9), foreground="#475569").pack(anchor="w", padx=12, pady=(2,0))

        self.chat_area = scrolledtext.ScrolledText(
            self.chat_tab, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#1e293b", fg="#e2e8f0", relief=tk.FLAT, padx=10, pady=10,
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.chat_area.config(state=tk.DISABLED)

        self.chat_area.tag_config("you",      foreground="#3b82f6", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("bot",      foreground="#22d3ee", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("text",     foreground="#e2e8f0", font=("Segoe UI", 11))
        self.chat_area.tag_config("positive", foreground="#22c55e", font=("Segoe UI", 9, "italic"))
        self.chat_area.tag_config("negative", foreground="#ef4444", font=("Segoe UI", 9, "italic"))
        self.chat_area.tag_config("neutral",  foreground="#94a3b8", font=("Segoe UI", 9, "italic"))
        self.chat_area.tag_config("src",      foreground="#475569", font=("Segoe UI", 9, "italic"))

        inp = ttk.Frame(self.chat_tab)
        inp.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.user_input = ttk.Entry(inp, font=("Segoe UI", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.user_input.bind("<Return>", lambda e: self.send_message())
        ttk.Button(inp, text="Send", command=self.send_message).pack(side=tk.RIGHT)

        btn = ttk.Frame(self.chat_tab)
        btn.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(btn, text="Clear", command=self.clear_chat).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(btn, text="Save Chat", command=self.save_chat).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(btn, text="Help", command=self.show_help).pack(side=tk.RIGHT)

    def _build_stats_tab(self):
        ttk.Label(self.stats_tab, text="Sentiment History",
                  font=("Segoe UI", 13, "bold")).pack(pady=(16, 4))

        # counters
        cf = ttk.Frame(self.stats_tab)
        cf.pack(fill=tk.X, padx=20, pady=8)

        self.pos_var = tk.StringVar(value="0")
        self.neg_var = tk.StringVar(value="0")
        self.neu_var = tk.StringVar(value="0")

        for lbl, var, col in [
            ("Positive", self.pos_var, "#22c55e"),
            ("Negative", self.neg_var, "#ef4444"),
            ("Neutral",  self.neu_var, "#94a3b8"),
        ]:
            box = tk.Frame(cf, bg="#1e293b")
            box.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
            tk.Label(box, textvariable=var, font=("Segoe UI", 32, "bold"),
                     bg="#1e293b", fg=col).pack(pady=(12, 4))
            tk.Label(box, text=lbl, font=("Segoe UI", 10),
                     bg="#1e293b", fg="#94a3b8").pack(pady=(0, 12))

        ttk.Separator(self.stats_tab, orient="horizontal").pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(self.stats_tab, text="Message Log",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20)

        self.log_text = scrolledtext.ScrolledText(
            self.stats_tab, wrap=tk.WORD, height=14,
            font=("Segoe UI", 10), bg="#1e293b", fg="#e2e8f0",
            relief=tk.FLAT, padx=8, pady=6,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)
        self.log_text.config(state=tk.DISABLED)
        self.log_text.tag_config("pos", foreground="#22c55e")
        self.log_text.tag_config("neg", foreground="#ef4444")
        self.log_text.tag_config("neu", foreground="#94a3b8")

    def _build_api_tab(self):
        ttk.Label(self.api_tab, text="Gemini API Key",
                  font=("Segoe UI", 12, "bold")).pack(pady=(20, 6))
        ttk.Label(self.api_tab, text="Optional - used when Keras model is not confident",
                  font=("Segoe UI", 9), foreground="#475569").pack()

        self.api_entry = ttk.Entry(self.api_tab, width=60, show="*")
        self.api_entry.pack(padx=20, pady=10)

        self.api_status = tk.StringVar(value="")
        ttk.Button(self.api_tab, text="Save", command=self._save_key).pack()
        ttk.Label(self.api_tab, textvariable=self.api_status,
                  foreground="#22c55e").pack(pady=4)

        ttk.Separator(self.api_tab, orient="horizontal").pack(fill=tk.X, padx=20, pady=12)
        ttk.Label(self.api_tab, text="Get key at: aistudio.google.com/apikey",
                  foreground="#475569").pack()

    def _save_key(self):
        k = self.api_entry.get().strip()
        if k:
            self.gemini_key = k
            self.api_status.set("Key saved!")
        else:
            messagebox.showwarning("Missing", "Please enter a key.")

    def load_data(self):
        if KERAS_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                self.intents = json.loads(open(INTENTS_PATH, encoding="utf-8").read())
                self.words   = pickle.load(open(WORDS_PATH, "rb"))
                self.classes = pickle.load(open(CLASSES_PATH, "rb"))
                self.model   = load_model(MODEL_PATH)
                self.keras_ready = True
                self.status_var.set("Model loaded - sentiment analysis active")
            except Exception as e:
                print("load error:", e)
                self.keras_ready = False
                self.status_var.set("Model not found")
        else:
            self.keras_ready = False
            self.status_var.set("Add Gemini key to use chatbot")

        self._add_msg("bot", "Bot",
            "Hello! I am your fitness assistant with sentiment detection.\n"
            "I can tell if you are happy, sad or neutral from your messages!")

    def analyse_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= POS_THRESHOLD:
            return "positive", compound
        elif compound <= NEG_THRESHOLD:
            return "negative", compound
        else:
            return "neutral", compound

    def send_message(self):
        msg = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        if not msg:
            return

        # detect sentiment before replying
        sentiment, compound = self.analyse_sentiment(msg)

        # update the indicator at top
        icons = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
        colors = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}
        self.sentiment_var.set(icons[sentiment])
        self.sent_label.config(fg=colors[sentiment])
        self.score_var.set(f"(score: {compound})")

        self._add_msg("you", "You", msg)

        # show sentiment below user message
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"  [{sentiment} | score: {compound}]\n", sentiment)
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

        # save to log
        self.sentiment_log.append({
            "msg": msg, "sentiment": sentiment,
            "compound": compound,
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        })
        self._update_stats()

        # get response in background thread so UI doesnt freeze
        def get_resp():
            resp, src = self._get_response(msg, sentiment)
            self.master.after(0, lambda: self._add_msg("bot", "Bot", resp, src))
            self.conversation_history.append((msg, resp))

        threading.Thread(target=get_resp, daemon=True).start()

    def _get_response(self, msg, sentiment):
        low = msg.lower()

        if low in ("exit", "quit", "bye"):
            return "Goodbye! Stay healthy!", "built-in"
        if low == "time":
            return f"Current time: {datetime.datetime.now().strftime('%H:%M:%S')}", "built-in"

        # keras model first
        if self.keras_ready:
            ints = self._predict(msg)
            if ints and float(ints[0]["probability"]) >= THRESHOLD:
                resp = self._keras_resp(ints)
                if resp:
                    # add empathy for negative messages
                    if sentiment == "negative":
                        resp = "I understand that can be tough. " + resp
                    return resp, f"Keras ({float(ints[0]['probability']):.0%})"

        # gemini fallback
        if GEMINI_AVAILABLE and self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                tone_map = {
                    "positive": "enthusiastic and encouraging",
                    "negative": "empathetic and supportive",
                    "neutral": "informative",
                }
                prompt = (
                    f"You are a fitness assistant. The user seems {sentiment}. "
                    f"Reply in a {tone_map[sentiment]} tone.\n"
                    f"User: {msg}\nAssistant:"
                )
                r = model.generate_content(prompt)
                return r.text, f"Gemini ({sentiment} tone)"
            except Exception as e:
                print("gemini error:", e)

        return "I'm not sure. Please try rephrasing!", "–"

    def _predict(self, sentence):
        words_in = [self.lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]
        bow = [1 if w in words_in else 0 for w in self.words]
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        results = [[i, float(r)] for i, r in enumerate(res) if r > THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]

    def _keras_resp(self, intents_list):
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return None

    def _update_stats(self):
        pos = sum(1 for s in self.sentiment_log if s["sentiment"] == "positive")
        neg = sum(1 for s in self.sentiment_log if s["sentiment"] == "negative")
        neu = sum(1 for s in self.sentiment_log if s["sentiment"] == "neutral")
        self.pos_var.set(str(pos))
        self.neg_var.set(str(neg))
        self.neu_var.set(str(neu))

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        for entry in reversed(self.sentiment_log[-20:]):
            tag = entry["sentiment"][:3]
            icon = {"positive": "😊", "negative": "😟", "neutral": "😐"}[entry["sentiment"]]
            self.log_text.insert(tk.END,
                f"[{entry['time']}] {icon} {entry['sentiment'].upper()} ({entry['compound']})\n", tag)
            self.log_text.insert(tk.END,
                f"  \"{entry['msg'][:60]}{'...' if len(entry['msg'])>60 else ''}\"\n\n")
        self.log_text.config(state=tk.DISABLED)

    def _add_msg(self, role, label, text, source=""):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{label}: ", role)
        self.chat_area.insert(tk.END, text + "\n", "text")
        if source:
            self.chat_area.insert(tk.END, f"  source: {source}\n", "src")
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def clear_chat(self):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.conversation_history.clear()

    def save_chat(self):
        fname = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for u, b in self.conversation_history:
                f.write(f"You: {u}\nBot: {b}\n\n")
        messagebox.showinfo("Saved", f"Saved to {fname}")

    def show_help(self):
        messagebox.showinfo("Help", (
            "Fitness Chatbot with Sentiment Analysis\n\n"
            "Every message is analysed using NLTK VADER.\n"
            "Positive messages get enthusiastic replies.\n"
            "Negative messages get empathetic replies.\n\n"
            "Check Sentiment Stats tab to see history.\n\n"
            "Commands: exit/quit/bye, time"
        ))


if __name__ == "__main__":
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("vader_lexicon", quiet=True)

    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()
