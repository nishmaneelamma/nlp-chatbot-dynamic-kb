import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import datetime
import os

# tried to import these, hope they work
try:
    from keras.models import load_model
    keras_ok = True
except:
    keras_ok = False

try:
    from googletrans import Translator, LANGUAGES
    translate_ok = True
except:
    translate_ok = False

try:
    import google.generativeai as genai
    gemini_ok = True
except:
    gemini_ok = False

# file paths
BASE = os.path.dirname(os.path.abspath(__file__))
INTENTS = os.path.join(BASE, "intents.json")
WORDS = os.path.join(BASE, "words.pkl")
CLASSES = os.path.join(BASE, "classes.pkl")
MODEL = os.path.join(BASE, "chatbot_model.h5")

THRESHOLD = 0.25

# languages supported
# english is default, added hindi, spanish, french, german
SUPPORTED_LANGS = {
    "English":  "en",
    "Hindi":    "hi",
    "Spanish":  "es",
    "French":   "fr",
    "German":   "de",
}

# some greetings in each language for cultural touch
GREETINGS = {
    "en": "Hello! How can I help you today?",
    "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?",
    "es": "¡Hola! ¿Cómo puedo ayudarte hoy?",
    "fr": "Bonjour! Comment puis-je vous aider aujourd'hui?",
    "de": "Hallo! Wie kann ich Ihnen heute helfen?",
}

SORRY = {
    "en": "Sorry, I didn't understand that.",
    "hi": "माफ करें, मैं यह नहीं समझ सका।",
    "es": "Lo siento, no entendí eso.",
    "fr": "Désolé, je n'ai pas compris.",
    "de": "Entschuldigung, das habe ich nicht verstanden.",
}


class MultilingualChatbot:
    def __init__(self, master):
        self.master = master
        self.lemmatizer = WordNetLemmatizer()
        self.translator = Translator() if translate_ok else None
        self.current_lang = "en"
        self.detected_lang = "en"
        self.gemini_key = ""
        self.chat_log = []

        self.build_gui()
        self.load_model_data()

    def build_gui(self):
        self.master.title("Multilingual Fitness Chatbot")
        self.master.geometry("900x650")
        self.master.configure(bg="#1a1a2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("TButton", background="#16213e", foreground="#e0e0e0",
                        borderwidth=0, padding=6)
        style.map("TButton",
                  background=[("active", "#0f3460")],
                  foreground=[("active", "#e94560")])
        style.configure("TEntry", fieldbackground="#16213e",
                        foreground="#e0e0e0", insertcolor="#e0e0e0")
        style.configure("TCombobox", fieldbackground="#16213e",
                        foreground="#e0e0e0", background="#16213e")
        style.configure("TNotebook", background="#1a1a2e")
        style.configure("TNotebook.Tab", background="#16213e",
                        foreground="#aaaaaa", padding=[12, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", "#0f3460")],
                  foreground=[("selected", "#e94560")])

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_tab = ttk.Frame(self.notebook)
        self.lang_tab = ttk.Frame(self.notebook)
        self.key_tab  = ttk.Frame(self.notebook)

        self.notebook.add(self.chat_tab, text="Chat")
        self.notebook.add(self.lang_tab, text="Language Settings")
        self.notebook.add(self.key_tab,  text="API Key")

        self._chat_tab()
        self._lang_tab()
        self._key_tab()

    def _chat_tab(self):
        # top bar showing detected and selected language
        top = ttk.Frame(self.chat_tab)
        top.pack(fill=tk.X, padx=12, pady=(8, 2))

        ttk.Label(top, text="Detected Language:").pack(side=tk.LEFT)
        self.detected_var = tk.StringVar(value="English")
        ttk.Label(top, textvariable=self.detected_var,
                  font=("Segoe UI", 9, "bold"),
                  foreground="#e94560").pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="|").pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Responding in:").pack(side=tk.LEFT)
        self.responding_var = tk.StringVar(value="English")
        ttk.Label(top, textvariable=self.responding_var,
                  font=("Segoe UI", 9, "bold"),
                  foreground="#00b4d8").pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Loading...")
        ttk.Label(self.chat_tab, textvariable=self.status_var,
                  font=("Segoe UI", 9),
                  foreground="#555555").pack(anchor="w", padx=12)

        # chat area
        self.chat_area = scrolledtext.ScrolledText(
            self.chat_tab, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#16213e", fg="#e0e0e0", insertbackground="#e0e0e0",
            relief=tk.FLAT, padx=10, pady=10,
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.chat_area.config(state=tk.DISABLED)

        self.chat_area.tag_config("you",  foreground="#e94560",
                                  font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("bot",  foreground="#00b4d8",
                                  font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("text", foreground="#e0e0e0",
                                  font=("Segoe UI", 11))
        self.chat_area.tag_config("info", foreground="#666666",
                                  font=("Segoe UI", 9, "italic"))

        # input
        inp = ttk.Frame(self.chat_tab)
        inp.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.msg_input = ttk.Entry(inp, font=("Segoe UI", 11))
        self.msg_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.msg_input.bind("<Return>", lambda e: self.send())
        ttk.Button(inp, text="Send", command=self.send).pack(side=tk.RIGHT)

        btns = ttk.Frame(self.chat_tab)
        btns.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(btns, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btns, text="Save", command=self.save).pack(side=tk.LEFT)
        ttk.Button(btns, text="Help", command=self.help).pack(side=tk.RIGHT)

    def _lang_tab(self):
        ttk.Label(self.lang_tab, text="Language Settings",
                  font=("Segoe UI", 13, "bold")).pack(pady=(20, 8))

        # auto detect toggle
        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.lang_tab,
                        text="Auto-detect language from my messages",
                        variable=self.auto_detect_var).pack(pady=6)

        ttk.Separator(self.lang_tab, orient="horizontal").pack(fill=tk.X, padx=20, pady=10)

        # manual language select
        ttk.Label(self.lang_tab,
                  text="Or manually select response language:").pack()
        self.lang_combo = ttk.Combobox(self.lang_tab,
                                        values=list(SUPPORTED_LANGS.keys()),
                                        state="readonly", width=20)
        self.lang_combo.set("English")
        self.lang_combo.pack(pady=8)
        ttk.Button(self.lang_tab, text="Apply Language",
                   command=self.apply_lang).pack()

        ttk.Separator(self.lang_tab, orient="horizontal").pack(fill=tk.X, padx=20, pady=12)

        # supported languages list
        ttk.Label(self.lang_tab,
                  text="Supported Languages:",
                  font=("Segoe UI", 11, "bold")).pack()

        for name, code in SUPPORTED_LANGS.items():
            greeting = GREETINGS.get(code, "")
            ttk.Label(self.lang_tab,
                      text=f"{name} ({code})  —  {greeting}",
                      font=("Segoe UI", 10),
                      foreground="#888888").pack(pady=2)

    def _key_tab(self):
        ttk.Label(self.key_tab, text="Gemini API Key",
                  font=("Segoe UI", 12, "bold")).pack(pady=(20, 6))
        ttk.Label(self.key_tab,
                  text="Used for answering questions not in the trained dataset.",
                  font=("Segoe UI", 9),
                  foreground="#555555").pack()

        self.key_entry = ttk.Entry(self.key_tab, width=55, show="*")
        self.key_entry.pack(padx=20, pady=10)

        self.key_status = tk.StringVar(value="")
        ttk.Button(self.key_tab, text="Save Key",
                   command=self._save_key).pack()
        ttk.Label(self.key_tab, textvariable=self.key_status,
                  foreground="#00b4d8").pack(pady=4)

        ttk.Separator(self.key_tab, orient="horizontal").pack(fill=tk.X, padx=20, pady=12)
        ttk.Label(self.key_tab,
                  text="Get a free key at: aistudio.google.com/apikey",
                  font=("Segoe UI", 9),
                  foreground="#555555").pack()

    def _save_key(self):
        k = self.key_entry.get().strip()
        if k:
            self.gemini_key = k
            self.key_status.set("Key saved!")
        else:
            messagebox.showwarning("Missing", "Please enter a key.")

    def load_model_data(self):
        if keras_ok and os.path.exists(MODEL):
            try:
                self.intents = json.loads(open(INTENTS, encoding="utf-8").read())
                self.words   = pickle.load(open(WORDS, "rb"))
                self.classes = pickle.load(open(CLASSES, "rb"))
                self.model   = load_model(MODEL)
                self.keras_ready = True
                self.status_var.set("Model loaded OK")
            except Exception as e:
                print("model load error:", e)
                self.keras_ready = False
                self.status_var.set("Model not found, using Gemini only")
        else:
            self.keras_ready = False
            if os.path.exists(INTENTS):
                self.intents = json.loads(open(INTENTS, encoding="utf-8").read())
            self.status_var.set("Add Gemini key to start chatting")

        self._add_msg("bot", "Bot", GREETINGS["en"])
        self._add_msg("bot", "Bot",
                      "I can chat in English, Hindi, Spanish, French and German!\n"
                      "Just type in any language and I will detect it automatically.")

    def apply_lang(self):
        chosen = self.lang_combo.get()
        self.current_lang = SUPPORTED_LANGS[chosen]
        self.responding_var.set(chosen)
        greeting = GREETINGS.get(self.current_lang, GREETINGS["en"])
        self._add_msg("bot", "Bot", f"Language set to {chosen}. {greeting}")

    def detect_language(self, text):
        # use googletrans to detect what language the user typed
        if not translate_ok or not self.translator:
            return "en"
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            print("detect error:", e)
            return "en"

    def translate_text(self, text, src, dest):
        # translate text from src language to dest language
        if not translate_ok or not self.translator:
            return text
        if src == dest:
            return text
        try:
            result = self.translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            print("translate error:", e)
            return text

    def send(self):
        msg = self.msg_input.get().strip()
        self.msg_input.delete(0, tk.END)
        if not msg:
            return

        self._add_msg("you", "You", msg)

        # step 1: detect language of user message
        if self.auto_detect_var.get():
            detected = self.detect_language(msg)
            self.detected_lang = detected
            lang_name = LANGUAGES.get(detected, detected).title() if translate_ok else "English"
            self.detected_var.set(lang_name)
            self._add_msg("bot", "", f"(Detected: {lang_name})", tag="info")

            # respond in detected language unless user changed it manually
            respond_in = detected
            self.current_lang = detected
            lang_display = LANGUAGES.get(detected, detected).title() if translate_ok else "English"
            self.responding_var.set(lang_display)
        else:
            detected = "en"
            respond_in = self.current_lang

        # step 2: translate message to english for the keras model
        msg_in_english = self.translate_text(msg, detected, "en")

        # step 3: get response in english
        response_en, source = self.get_response(msg_in_english)

        # step 4: translate response back to user's language
        final_response = self.translate_text(response_en, "en", respond_in)

        self._add_msg("bot", "Bot", final_response, source)
        self.chat_log.append((msg, final_response))

    def get_response(self, msg):
        low = msg.lower()

        if low in ("exit", "quit", "bye"):
            return "Goodbye! Stay healthy!", "built-in"
        if low == "time":
            return f"Current time: {datetime.datetime.now().strftime('%H:%M:%S')}", "built-in"

        # try keras model first
        if self.keras_ready:
            try:
                ints = self.predict(msg)
                if ints and float(ints[0]["probability"]) >= THRESHOLD:
                    resp = self.keras_response(ints)
                    if resp:
                        return resp, f"Keras ({float(ints[0]['probability']):.0%})"
            except Exception as e:
                print("keras predict error:", e)

        # gemini fallback
        if gemini_ok and self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                m = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    "You are a helpful fitness and health assistant. "
                    "Answer this question clearly and briefly: " + msg
                )
                r = m.generate_content(prompt)
                return r.text, "Gemini"
            except Exception as e:
                print("gemini error:", e)

        return "Sorry I did not understand that. Try rephrasing.", "–"

    def predict(self, sentence):
        words_in_sentence = [
            self.lemmatizer.lemmatize(w.lower())
            for w in nltk.word_tokenize(sentence)
        ]
        bow = [1 if w in words_in_sentence else 0 for w in self.words]
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        results = [[i, float(r)] for i, r in enumerate(res) if r > THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]

    def keras_response(self, intents_list):
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return None

    def _add_msg(self, role, label, text, source="", tag=None):
        self.chat_area.config(state=tk.NORMAL)
        if tag:
            if label:
                self.chat_area.insert(tk.END, label + ": ", role)
            self.chat_area.insert(tk.END, text + "\n", tag)
        else:
            if label:
                self.chat_area.insert(tk.END, label + ": ", role)
            self.chat_area.insert(tk.END, text + "\n", "text")
            if source:
                self.chat_area.insert(tk.END, f"  source: {source}\n", "info")
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def clear(self):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_log.clear()

    def save(self):
        fname = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for u, b in self.chat_log:
                f.write(f"You: {u}\nBot: {b}\n\n")
        messagebox.showinfo("Saved", f"Saved to {fname}")

    def help(self):
        messagebox.showinfo("Help", (
            "Multilingual Fitness Chatbot\n\n"
            "Supported languages:\n"
            "  English, Hindi, Spanish, French, German\n\n"
            "How to use:\n"
            "  Just type in any language.\n"
            "  The bot detects it automatically and replies\n"
            "  in the same language.\n\n"
            "Or go to Language Settings tab to\n"
            "manually pick a language.\n\n"
            "Special commands:\n"
            "  bye / exit / quit\n"
            "  time"
        ))


if __name__ == "__main__":
    nltk.download("punkt",     quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("punkt_tab", quiet=True)

    root = tk.Tk()
    MultilingualChatbot(root)
    root.mainloop()
