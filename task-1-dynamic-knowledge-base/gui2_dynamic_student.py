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
import threading
import time
import hashlib

try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except:
    CHROMA_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
WORDS_PATH = os.path.join(BASE_DIR, "words.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.h5")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

ERROR_THRESHOLD = 0.25
FALLBACK_THRESHOLD = 0.45
AUTO_REFRESH_SECS = 60

print("imports done")


class DynamicKnowledgeBase:
    # this class handles all the chromadb stuff
    # chromadb stores text as vectors so we can search by meaning

    def __init__(self):
        if not CHROMA_AVAILABLE:
            self.collection = None
            return

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="chatbot_knowledge",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        print("chroma collection ready, count:", self.collection.count())

    def add_entry(self, tag, pattern, response):
        if self.collection is None:
            return False
        # use md5 hash as unique id so we dont add duplicates
        doc_id = hashlib.md5(f"{tag}:{pattern}".encode()).hexdigest()[:16]
        existing = self.collection.get(ids=[doc_id])
        if existing["ids"]:
            return False
        self.collection.add(
            ids=[doc_id],
            documents=[pattern],
            metadatas=[{"tag": tag, "response": response}],
        )
        return True

    def seed_from_intents(self, intents_path):
        if self.collection is None:
            return 0
        try:
            with open(intents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            return 0

        added = 0
        for intent in data.get("intents", []):
            tag = intent["tag"]
            responses = intent.get("responses", ["I'm not sure."])
            for pattern in intent.get("patterns", []):
                resp = random.choice(responses)
                if self.add_entry(tag, pattern, resp):
                    added += 1
        print(f"seeded {added} entries from intents.json")
        return added

    def search(self, query, n=3):
        if self.collection is None or self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n, self.collection.count()),
        )
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({
                "tag": meta.get("tag", "unknown"),
                "pattern": doc,
                "response": meta.get("response", ""),
                "distance": dist,
            })
        return out

    def count(self):
        if self.collection is None:
            return 0
        return self.collection.count()

    def list_entries(self, limit=100):
        if self.collection is None or self.collection.count() == 0:
            return [], []
        result = self.collection.get(limit=limit, include=["documents", "metadatas"])
        return result["documents"], result["metadatas"]


class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.lemmatizer = WordNetLemmatizer()
        self.conversation_history = []
        self._intents_mtime = 0

        self.setup_gui()
        self.load_chatbot_data()
        self._start_auto_refresh()

    def setup_gui(self):
        self.master.title("Fitness Chatbot with Dynamic Knowledge Base")
        self.master.geometry("860x640")
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
        style.configure("TNotebook", background="#1a1a2e")
        style.configure("TNotebook.Tab", background="#16213e",
                        foreground="#aaaaaa", padding=[14, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", "#0f3460")],
                  foreground=[("selected", "#e94560")])

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_tab = ttk.Frame(self.notebook)
        self.kb_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_tab, text="Chat")
        self.notebook.add(self.kb_tab, text="Knowledge Base")

        self._build_chat_tab()
        self._build_kb_tab()

    def _build_chat_tab(self):
        self.status_var = tk.StringVar(value="Loading...")
        ttk.Label(self.chat_tab, textvariable=self.status_var,
                  font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(6, 0))

        self.chat_history = scrolledtext.ScrolledText(
            self.chat_tab, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#16213e", fg="#e0e0e0", insertbackground="#e0e0e0",
            relief=tk.FLAT, padx=10, pady=10,
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.tag_config("you", foreground="#e94560", font=("Segoe UI", 11, "bold"))
        self.chat_history.tag_config("bot", foreground="#00b4d8", font=("Segoe UI", 11, "bold"))
        self.chat_history.tag_config("text", foreground="#e0e0e0", font=("Segoe UI", 11))
        self.chat_history.tag_config("src", foreground="#888888", font=("Segoe UI", 9, "italic"))

        inp_frame = ttk.Frame(self.chat_tab)
        inp_frame.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.user_input = ttk.Entry(inp_frame, font=("Segoe UI", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.user_input.bind("<Return>", lambda e: self.send_message())
        ttk.Button(inp_frame, text="Send", command=self.send_message).pack(side=tk.RIGHT)

        btn_frame = ttk.Frame(self.chat_tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(btn_frame, text="Clear", command=self.clear_chat).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Save Chat", command=self.save_chat).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Help", command=self.show_help).pack(side=tk.RIGHT)

    def _build_kb_tab(self):
        stats_frame = ttk.Frame(self.kb_tab)
        stats_frame.pack(fill=tk.X, padx=12, pady=(10, 4))
        self.kb_count_var = tk.StringVar(value="Documents in KB: 0")
        ttk.Label(stats_frame, textvariable=self.kb_count_var,
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.last_update_var = tk.StringVar(value="Last update: never")
        ttk.Label(stats_frame, textvariable=self.last_update_var,
                  font=("Segoe UI", 9), foreground="#888888").pack(side=tk.RIGHT)

        ttk.Separator(self.kb_tab, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        add_frame = ttk.LabelFrame(self.kb_tab, text="  Add New Entry  ", padding=10)
        add_frame.pack(fill=tk.X, padx=12, pady=4)

        ttk.Label(add_frame, text="Tag:").grid(row=0, column=0, sticky="w", pady=2)
        self.kb_tag = ttk.Entry(add_frame, width=30)
        self.kb_tag.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)

        ttk.Label(add_frame, text="Pattern:").grid(row=1, column=0, sticky="w", pady=2)
        self.kb_pattern = ttk.Entry(add_frame, width=60)
        self.kb_pattern.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)

        ttk.Label(add_frame, text="Response:").grid(row=2, column=0, sticky="nw", pady=2)
        self.kb_response = tk.Text(add_frame, height=4, width=60,
                                   bg="#16213e", fg="#e0e0e0",
                                   insertbackground="#e0e0e0", font=("Segoe UI", 10),
                                   relief=tk.FLAT, padx=6, pady=4)
        self.kb_response.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=2)
        add_frame.columnconfigure(1, weight=1)

        btn_row = ttk.Frame(add_frame)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="e", pady=(6, 0))
        ttk.Button(btn_row, text="Save Entry", command=self._add_kb_entry).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_row, text="Seed from intents.json", command=self._seed_from_intents).pack(side=tk.RIGHT)

        ttk.Separator(self.kb_tab, orient="horizontal").pack(fill=tk.X, padx=12, pady=6)

        browse_frame = ttk.LabelFrame(self.kb_tab, text="  Existing Entries  ", padding=6)
        browse_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        cols = ("Tag", "Pattern", "Response")
        self.kb_tree = ttk.Treeview(browse_frame, columns=cols, show="headings", height=10)
        for col, w in zip(cols, (120, 280, 320)):
            self.kb_tree.heading(col, text=col)
            self.kb_tree.column(col, width=w, anchor="w")
        vsb = ttk.Scrollbar(browse_frame, orient="vertical", command=self.kb_tree.yview)
        self.kb_tree.configure(yscrollcommand=vsb.set)
        self.kb_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(self.kb_tab, text="Refresh List", command=self._refresh_kb_list).pack(pady=(0, 6))

    def load_chatbot_data(self):
        if KERAS_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                self.intents = json.loads(open(INTENTS_PATH, encoding="utf-8").read())
                self.words = pickle.load(open(WORDS_PATH, "rb"))
                self.classes = pickle.load(open(CLASSES_PATH, "rb"))
                self.model = load_model(MODEL_PATH)
                self.keras_ready = True
            except Exception as e:
                print("error loading model:", e)
                self.keras_ready = False
                messagebox.showwarning("Model Error", f"Could not load model:\n{e}")
        else:
            self.keras_ready = False
            if os.path.exists(INTENTS_PATH):
                self.intents = json.loads(open(INTENTS_PATH, encoding="utf-8").read())

        # setup knowledge base
        self.kb = DynamicKnowledgeBase()
        if not CHROMA_AVAILABLE:
            messagebox.showwarning("Missing Library", "Install chromadb:\n  pip install chromadb")

        # seed KB from intents if empty
        if CHROMA_AVAILABLE and self.kb.count() == 0:
            n = self.kb.seed_from_intents(INTENTS_PATH)
            self.status_var.set(f"Seeded {n} entries from intents.json")
        else:
            self.status_var.set(f"Ready | KB has {self.kb.count()} documents")

        self._refresh_kb_stats()
        self._refresh_kb_list()

        self._append_chat("bot", "Bot",
            "Hello! I am your Fitness & First-Aid assistant.\n"
            "I now have a dynamic knowledge base - you can add new info anytime!")

    def send_message(self):
        msg = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        if not msg:
            return
        self._append_chat("you", "You", msg)
        response, source = self.get_bot_response(msg)
        self._append_chat("bot", "Bot", response, source)
        self.conversation_history.append((msg, response))

    def get_bot_response(self, msg):
        low = msg.lower()

        if low in ("exit", "quit", "bye"):
            return "Goodbye! Stay healthy!", "built-in"
        if low == "time":
            return f"Current time: {datetime.datetime.now().strftime('%H:%M:%S')}", "built-in"
        if low.startswith("search "):
            import webbrowser
            q = msg[7:]
            webbrowser.open(f"https://www.google.com/search?q={q}")
            return f"Opened web search for '{q}'.", "built-in"

        best_response = None
        best_confidence = 0.0
        source = "unknown"

        # try keras model
        if self.keras_ready:
            ints = self.predict_class(msg)
            if ints:
                best_confidence = float(ints[0]["probability"])
                if best_confidence >= ERROR_THRESHOLD:
                    best_response = self.get_keras_response(ints)
                    source = f"Keras model ({best_confidence:.0%})"

        # try knowledge base as fallback
        if best_response is None or best_confidence < FALLBACK_THRESHOLD:
            kb_hits = self.kb.search(msg, n=1)
            if kb_hits:
                hit = kb_hits[0]
                kb_sim = 1 - hit["distance"]
                if kb_sim > best_confidence:
                    best_response = hit["response"]
                    source = f"Knowledge Base ({kb_sim:.0%} match)"

        if best_response:
            return best_response, source

        return "I'm not sure about that. You can add info to the Knowledge Base tab!", "–"

    def clean_up_sentence(self, sentence):
        return [self.lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

    def bag_of_words(self, sentence):
        sw = self.clean_up_sentence(sentence)
        return np.array([1 if w in sw else 0 for w in self.words])

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        results = sorted([[i, float(r)] for i, r in enumerate(res) if r > ERROR_THRESHOLD],
                         key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]

    def get_keras_response(self, intents_list):
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return None

    def _add_kb_entry(self):
        tag = self.kb_tag.get().strip()
        pattern = self.kb_pattern.get().strip()
        response = self.kb_response.get("1.0", tk.END).strip()
        if not tag or not pattern or not response:
            messagebox.showwarning("Missing", "Please fill in all fields.")
            return
        added = self.kb.add_entry(tag, pattern, response)
        if added:
            messagebox.showinfo("Done", f"Entry added! KB now has {self.kb.count()} docs.")
            self.kb_tag.delete(0, tk.END)
            self.kb_pattern.delete(0, tk.END)
            self.kb_response.delete("1.0", tk.END)
        else:
            messagebox.showinfo("Duplicate", "This pattern already exists.")
        self._refresh_kb_stats()
        self._refresh_kb_list()

    def _seed_from_intents(self):
        n = self.kb.seed_from_intents(INTENTS_PATH)
        messagebox.showinfo("Done", f"Added {n} new entries.\nTotal: {self.kb.count()}")
        self._refresh_kb_stats()
        self._refresh_kb_list()

    def _refresh_kb_stats(self):
        self.kb_count_var.set(f"Documents in KB: {self.kb.count()}")
        self.last_update_var.set(f"Last update: {datetime.datetime.now().strftime('%H:%M:%S')}")

    def _refresh_kb_list(self):
        for row in self.kb_tree.get_children():
            self.kb_tree.delete(row)
        docs, metas = self.kb.list_entries()
        for doc, meta in zip(docs, metas):
            self.kb_tree.insert("", tk.END, values=(
                meta.get("tag", "–"),
                doc[:80],
                meta.get("response", "–")[:80],
            ))

    def _start_auto_refresh(self):
        # background thread that checks if intents.json changed and re-indexes
        def worker():
            while True:
                time.sleep(AUTO_REFRESH_SECS)
                try:
                    mtime = os.path.getmtime(INTENTS_PATH)
                    if mtime != self._intents_mtime:
                        self._intents_mtime = mtime
                        n = self.kb.seed_from_intents(INTENTS_PATH)
                        if n:
                            self.master.after(0, lambda: self.status_var.set(
                                f"Auto-refreshed: added {n} new entries"))
                            self.master.after(0, self._refresh_kb_stats)
                            self.master.after(0, self._refresh_kb_list)
                except Exception as e:
                    print("auto refresh error:", e)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _append_chat(self, role, label, text, source=""):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"{label}: ", role)
        self.chat_history.insert(tk.END, text + "\n", "text")
        if source:
            self.chat_history.insert(tk.END, f"  source: {source}\n", "src")
        self.chat_history.insert(tk.END, "\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.conversation_history.clear()

    def save_chat(self):
        fname = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for u, b in self.conversation_history:
                f.write(f"You: {u}\nBot: {b}\n\n")
        messagebox.showinfo("Saved", f"Saved to {fname}")

    def show_help(self):
        messagebox.showinfo("Help", (
            "Fitness & First-Aid Chatbot with Dynamic Knowledge Base\n\n"
            "Knowledge Base tab:\n"
            "  Add new topics without retraining\n"
            "  Seed all intents from intents.json\n"
            "  Auto-refreshes if intents.json changes\n\n"
            "Each reply shows its source:\n"
            "  Keras model or Knowledge Base\n\n"
            "Special commands:\n"
            "  exit/quit/bye, time, search <query>"
        ))


if __name__ == "__main__":
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()
