import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, filedialog
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import datetime
import os
import threading

# PIL for showing images in tkinter
try:
    from PIL import Image, ImageTk
    pil_ok = True
except:
    pil_ok = False
    print("PIL not found, install with: pip install Pillow")

# keras for the original chatbot model
try:
    from keras.models import load_model
    keras_ok = True
except:
    keras_ok = False
    print("keras not found")

# gemini for image analysis and text fallback
try:
    import google.generativeai as genai
    gemini_ok = True
except:
    gemini_ok = False
    print("google-generativeai not found")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS = os.path.join(BASE_DIR, "intents.json")
WORDS = os.path.join(BASE_DIR, "words.pkl")
CLASSES = os.path.join(BASE_DIR, "classes.pkl")
MODEL = os.path.join(BASE_DIR, "chatbot_model.h5")

THRESHOLD = 0.25


class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.lemmatizer = WordNetLemmatizer()
        self.chat_log = []
        self.current_image = None
        self.current_image_path = None
        self.api_key = ""

        self.build_gui()
        self.load_data()

    def build_gui(self):
        self.master.title("Fitness Chatbot - Multimodal Edition")
        self.master.geometry("900x680")
        self.master.configure(bg="#1a1a2e")
        self.master.resizable(True, True)

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
                        foreground="#aaaaaa", padding=[12, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", "#0f3460")],
                  foreground=[("selected", "#e94560")])

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_tab  = ttk.Frame(self.notebook)
        self.img_tab   = ttk.Frame(self.notebook)
        self.setup_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.chat_tab,  text="Chat")
        self.notebook.add(self.img_tab,   text="Image Analysis")
        self.notebook.add(self.setup_tab, text="API Key Setup")

        self._build_chat_tab()
        self._build_image_tab()
        self._build_setup_tab()

    def _build_chat_tab(self):
        self.status_var = tk.StringVar(value="Loading...")
        ttk.Label(self.chat_tab, textvariable=self.status_var,
                  font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(6,0))

        self.chat_area = scrolledtext.ScrolledText(
            self.chat_tab, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#16213e", fg="#e0e0e0", insertbackground="#e0e0e0",
            relief=tk.FLAT, padx=10, pady=10,
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.chat_area.config(state=tk.DISABLED)

        self.chat_area.tag_config("you",  foreground="#e94560", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("bot",  foreground="#00b4d8", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("text", foreground="#e0e0e0", font=("Segoe UI", 11))
        self.chat_area.tag_config("src",  foreground="#888888", font=("Segoe UI", 9, "italic"))

        inp = ttk.Frame(self.chat_tab)
        inp.pack(fill=tk.X, padx=10, pady=(0,4))
        self.msg_input = ttk.Entry(inp, font=("Segoe UI", 11))
        self.msg_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,6))
        self.msg_input.bind("<Return>", lambda e: self.send_message())
        ttk.Button(inp, text="Send", command=self.send_message).pack(side=tk.RIGHT)

        btns = ttk.Frame(self.chat_tab)
        btns.pack(fill=tk.X, padx=10, pady=(0,8))
        ttk.Button(btns, text="Clear", command=self.clear_chat).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(btns, text="Save Chat", command=self.save_chat).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(btns, text="Help", command=self.show_help).pack(side=tk.RIGHT)

    def _build_image_tab(self):
        ttk.Label(self.img_tab,
                  text="Upload an image and ask a question about it. Uses Gemini Vision.",
                  font=("Segoe UI", 10), foreground="#888888").pack(pady=(10,4))

        # image preview box
        self.img_label = tk.Label(self.img_tab,
                                  text="No image uploaded yet.\nClick Upload Image to start.",
                                  bg="#16213e", fg="#888888",
                                  font=("Segoe UI", 10),
                                  width=60, height=12,
                                  relief=tk.FLAT)
        self.img_label.pack(fill=tk.X, padx=12)

        upload_frame = ttk.Frame(self.img_tab)
        upload_frame.pack(fill=tk.X, padx=12, pady=4)
        ttk.Button(upload_frame, text="Upload Image",
                   command=self.upload_image).pack(side=tk.LEFT, padx=(0,6))
        self.img_path_var = tk.StringVar(value="No file selected")
        ttk.Label(upload_frame, textvariable=self.img_path_var,
                  font=("Segoe UI", 9), foreground="#888888").pack(side=tk.LEFT)

        ttk.Separator(self.img_tab, orient="horizontal").pack(fill=tk.X, padx=12, pady=6)

        ttk.Label(self.img_tab,
                  text="Ask something about the image (optional):",
                  font=("Segoe UI", 10)).pack(anchor="w", padx=12)
        self.img_prompt = ttk.Entry(self.img_tab, font=("Segoe UI", 11))
        self.img_prompt.pack(fill=tk.X, padx=12, pady=4)
        self.img_prompt.insert(0, "What do you see in this image?")

        ttk.Button(self.img_tab, text="Analyse Image",
                   command=self.analyse_image).pack(pady=4)

        ttk.Separator(self.img_tab, orient="horizontal").pack(fill=tk.X, padx=12, pady=6)

        ttk.Label(self.img_tab, text="Gemini Response:",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12)

        self.img_response = scrolledtext.ScrolledText(
            self.img_tab, wrap=tk.WORD, height=10,
            font=("Segoe UI", 11), bg="#16213e", fg="#e0e0e0",
            relief=tk.FLAT, padx=10, pady=8,
        )
        self.img_response.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0,10))
        self.img_response.config(state=tk.DISABLED)

    def _build_setup_tab(self):
        ttk.Label(self.setup_tab,
                  text="Enter your Gemini API key to enable image analysis.",
                  font=("Segoe UI", 10), foreground="#888888").pack(pady=(20,8))

        ttk.Label(self.setup_tab, text="API Key:",
                  font=("Segoe UI", 11)).pack(anchor="w", padx=20)
        self.key_entry = ttk.Entry(self.setup_tab, font=("Segoe UI", 11),
                                   width=60, show="*")
        self.key_entry.pack(padx=20, pady=4, fill=tk.X)

        ttk.Button(self.setup_tab, text="Save and Connect",
                   command=self.save_key).pack(pady=8)

        self.key_status = tk.StringVar(value="Not connected")
        ttk.Label(self.setup_tab, textvariable=self.key_status,
                  font=("Segoe UI", 10), foreground="#e94560").pack()

        ttk.Separator(self.setup_tab, orient="horizontal").pack(fill=tk.X, padx=20, pady=16)
        ttk.Label(self.setup_tab,
                  text="How to get a free API key:\n"
                       "1. Go to aistudio.google.com/apikey\n"
                       "2. Sign in with Google\n"
                       "3. Click Create API key\n"
                       "4. Paste it above",
                  font=("Segoe UI", 10), foreground="#888888",
                  justify=tk.LEFT).pack(anchor="w", padx=20)

    def save_key(self):
        k = self.key_entry.get().strip()
        if not k:
            messagebox.showwarning("Missing", "Please enter your API key.")
            return
        self.api_key = k
        # test if it works
        try:
            genai.configure(api_key=k)
            self.key_status.set("Connected!")
            self.status_var.set("Ready - Keras model + Gemini Vision active")
            messagebox.showinfo("Success", "API key saved! Image analysis is now enabled.")
        except Exception as e:
            print("key error:", e)
            self.key_status.set("Connection failed. Check your key.")

    def load_data(self):
        if keras_ok and os.path.exists(MODEL):
            try:
                self.intents = json.loads(open(INTENTS, encoding="utf-8").read())
                self.words   = pickle.load(open(WORDS, "rb"))
                self.classes = pickle.load(open(CLASSES, "rb"))
                self.model   = load_model(MODEL)
                self.keras_ready = True
                self.status_var.set("Ready - add Gemini API key for image features")
                print("model loaded ok")
            except Exception as e:
                print("model load error:", e)
                self.keras_ready = False
                self.status_var.set("Keras model not found")
        else:
            self.keras_ready = False
            if os.path.exists(INTENTS):
                self.intents = json.loads(open(INTENTS, encoding="utf-8").read())
            self.status_var.set("Add Gemini API key to use the chatbot")

        self._add_msg("bot", "Bot",
            "Hello! I am your Fitness and First Aid assistant.\n"
            "You can ask text questions or go to Image Analysis tab to upload an image.")

    def upload_image(self):
        if not pil_ok:
            messagebox.showerror("Missing library", "Install Pillow: pip install Pillow")
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        try:
            self.current_image = Image.open(path)
            self.current_image_path = path
            self.img_path_var.set(os.path.basename(path))

            # show preview
            preview = self.current_image.copy()
            preview.thumbnail((500, 250))
            photo = ImageTk.PhotoImage(preview)
            self.img_label.configure(image=photo, text="")
            self.img_label.image = photo
            print("image loaded:", path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {e}")

    def analyse_image(self):
        if self.current_image is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        if not self.api_key:
            messagebox.showwarning("No API key",
                                   "Please enter your Gemini API key in the API Key Setup tab.")
            return

        prompt = self.img_prompt.get().strip()

        self.img_response.config(state=tk.NORMAL)
        self.img_response.delete("1.0", tk.END)
        self.img_response.insert(tk.END, "Analysing image, please wait...")
        self.img_response.config(state=tk.DISABLED)

        # run in thread so GUI doesnt freeze
        def run():
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                if prompt:
                    full_prompt = (
                        f"You are a helpful fitness and health assistant. "
                        f"The user has shared an image and asked: {prompt}. "
                        f"Analyse the image and respond helpfully."
                    )
                    response = model.generate_content([full_prompt, self.current_image])
                else:
                    response = model.generate_content([
                        "You are a helpful fitness and health assistant. "
                        "Describe what you see in this image and give any relevant health insights.",
                        self.current_image
                    ])

                result = response.text
                print("gemini response received")
            except Exception as e:
                result = f"Error analysing image: {e}"
                print("gemini error:", e)

            # update UI from main thread
            self.master.after(0, lambda: self._show_img_response(result))
            fname = os.path.basename(self.current_image_path) if self.current_image_path else "image"
            self.master.after(0, lambda: self._add_msg("you", "You", f"[Image: {fname}] {prompt}"))
            self.master.after(0, lambda: self._add_msg("bot", "Bot", result, "Gemini Vision"))

        threading.Thread(target=run, daemon=True).start()

    def _show_img_response(self, text):
        self.img_response.config(state=tk.NORMAL)
        self.img_response.delete("1.0", tk.END)
        self.img_response.insert(tk.END, text)
        self.img_response.config(state=tk.DISABLED)

    def send_message(self):
        msg = self.msg_input.get().strip()
        self.msg_input.delete(0, tk.END)
        if not msg:
            return
        self._add_msg("you", "You", msg)

        def run():
            resp, src = self.get_response(msg)
            self.master.after(0, lambda: self._add_msg("bot", "Bot", resp, src))
            self.chat_log.append((msg, resp))

        threading.Thread(target=run, daemon=True).start()

    def get_response(self, msg):
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

        # try keras first
        if self.keras_ready:
            ints = self.predict(msg)
            if ints and float(ints[0]["probability"]) >= THRESHOLD:
                resp = self.keras_response(ints)
                if resp:
                    return resp, f"Keras ({float(ints[0]['probability']):.0%})"

        # gemini fallback for text
        if gemini_ok and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    "You are a helpful fitness and first aid assistant. "
                    "Answer this question briefly: " + msg
                )
                r = model.generate_content(prompt)
                return r.text, "Gemini AI"
            except Exception as e:
                print("gemini text error:", e)

        return "I am not sure about that. Try the Image Analysis tab to upload an image!", "–"

    def predict(self, sentence):
        words_in = [self.lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]
        bow = [1 if w in words_in else 0 for w in self.words]
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
        self.chat_log.clear()

    def save_chat(self):
        fname = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for u, b in self.chat_log:
                f.write(f"You: {u}\nBot: {b}\n\n")
        messagebox.showinfo("Saved", f"Saved to {fname}")

    def show_help(self):
        messagebox.showinfo("Help", (
            "Fitness Chatbot - Multimodal Edition\n\n"
            "Chat tab:\n"
            "  Ask any fitness or first aid question.\n"
            "  Uses Keras model then Gemini as fallback.\n\n"
            "Image Analysis tab:\n"
            "  Upload any image and ask a question.\n"
            "  Powered by Gemini Vision.\n\n"
            "API Key Setup tab:\n"
            "  Enter Gemini API key to enable image features.\n"
            "  Get free key at aistudio.google.com/apikey\n\n"
            "Commands: exit/quit/bye, time, search <query>"
        ))


if __name__ == "__main__":
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()
