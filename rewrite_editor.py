import re
import math
import tkinter as tk
from tkinter import messagebox
from collections import Counter

INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
SEGMENT_SEPARATOR = "\n\n"
MIN_WORD_LENGTH = 4
GAP_PENALTY = -0.15
MIN_MATCH_SCORE = 0.12
MAX_GROUP = 3
FONT = ("DejaVu Sans", 12)
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig", "no", "inc", "ltd"}
SENTENCE_END = re.compile(r"([.!?]+[\"'\u201d\u2019)\]]*)(\s+)")
SUFFIXES = ("ingly", "ations", "ation", "ities", "ness", "ments", "ment", "ings", "ing", "ies", "ied", "ers", "ally", "ity", "ous", "ive", "ed", "es", "er", "ly", "s")

def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sentences = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        end = match.end(1)
        before = text[start:end].rstrip(".!?\"'\u201d\u2019)]")
        last_word = before.split(" ")[-1].lower().rstrip(".") if before else ""
        after = text[match.end():match.end() + 1]
        if last_word in ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
            continue
        if after and after.islower():
            continue
        sentences.append(text[start:end].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences

def stem(word):
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def tokens(sentence):
    words = re.findall(r"[^\W\d_]+", sentence.lower())
    return [stem(w) for w in words if len(w) >= MIN_WORD_LENGTH]

def build_idf(token_lists):
    doc_count = len(token_lists)
    frequency = Counter()
    for token_list in token_lists:
        frequency.update(set(token_list))
    return {t: math.log((1 + doc_count) / (1 + f)) + 1 for t, f in frequency.items()}

def vectorize(token_list, idf):
    counts = Counter(token_list)
    vector = {t: c * idf.get(t, 1.0) for t, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vector.values()))
    return vector, norm

def cosine(a, b):
    vec_a, norm_a = a
    vec_b, norm_b = b
    if not norm_a or not norm_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(v * vec_b.get(t, 0.0) for t, v in vec_a.items())
    return dot / (norm_a * norm_b)

def merge_vectors(token_lists, idf):
    merged = []
    for token_list in token_lists:
        merged.extend(token_list)
    return vectorize(merged, idf)

def align(upper, lower):
    upper_tokens = [tokens(s) for s in upper]
    lower_tokens = [tokens(s) for s in lower]
    idf = build_idf(upper_tokens + lower_tokens)
    n, m = len(upper), len(lower)
    cache = {}

    def score(i, di, j, dj):
        key = (i, di, j, dj)
        if key not in cache:
            a = merge_vectors(upper_tokens[i - di:i], idf)
            b = merge_vectors(lower_tokens[j - dj:j], idf)
            similarity = cosine(a, b)
            if similarity < MIN_MATCH_SCORE:
                cache[key] = None
            else:
                penalty = 0.05 * (di + dj - 2)
                cache[key] = similarity - penalty
        return cache[key]

    best = [[-1e18] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0 and best[i - 1][j] > -1e17:
                candidates.append((best[i - 1][j] + GAP_PENALTY, (1, 0)))
            if j > 0 and best[i][j - 1] > -1e17:
                candidates.append((best[i][j - 1] + GAP_PENALTY, (0, 1)))
            for di in range(1, min(MAX_GROUP, i) + 1):
                for dj in range(1, min(MAX_GROUP, j) + 1):
                    if di > 1 and dj > 1:
                        continue
                    if best[i - di][j - dj] < -1e17:
                        continue
                    s = score(i, di, j, dj)
                    if s is not None:
                        candidates.append((best[i - di][j - dj] + s, (di, dj)))
            if candidates:
                best[i][j], back[i][j] = max(candidates, key=lambda c: c[0])
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        di, dj = back[i][j]
        steps.append((list(range(i - di, i)), list(range(j - dj, j))))
        i, j = i - di, j - dj
    steps.reverse()
    return steps

def load_segments():
    with open(INPUT_FILE, encoding="utf-8") as handle:
        content = handle.read().replace("\r\n", "\n").strip()
    if SEGMENT_SEPARATOR not in content:
        raise ValueError("No blank line separator found in " + INPUT_FILE)
    upper_text, lower_text = content.split(SEGMENT_SEPARATOR, 1)
    return upper_text.strip(), lower_text.strip()

def build_steps():
    upper_text, lower_text = load_segments()
    upper = split_sentences(upper_text)
    lower = split_sentences(lower_text)
    steps = []
    for upper_idx, lower_idx in align(upper, lower):
        steps.append({
            "upper": " ".join(upper[k] for k in upper_idx),
            "lower": " ".join(lower[k] for k in lower_idx),
            "had_lower": bool(lower_idx),
            "had_upper": bool(upper_idx),
        })
    return steps

class Editor:
    def __init__(self, root, steps):
        self.root = root
        self.steps = steps
        self.index = 0
        root.title("Rewrite editor")
        root.geometry("900x560")
        self.status = tk.Label(root, anchor="w", font=FONT)
        self.status.pack(fill="x", padx=10, pady=(8, 2))
        self.upper_box = tk.Text(root, height=8, wrap="word", font=FONT, bg="#eeeeee", state="disabled")
        self.upper_box.pack(fill="both", expand=True, padx=10, pady=4)
        self.note = tk.Label(root, anchor="w", font=("DejaVu Sans", 10), fg="#555555")
        self.note.pack(fill="x", padx=10)
        self.lower_box = tk.Text(root, height=8, wrap="word", font=FONT, undo=True)
        self.lower_box.pack(fill="both", expand=True, padx=10, pady=4)
        bar = tk.Frame(root)
        bar.pack(fill="x", padx=10, pady=8)
        tk.Button(bar, text="Back (Alt+Left)", command=self.back).pack(side="left")
        tk.Button(bar, text="Next (Alt+Right)", command=self.forward).pack(side="left", padx=6)
        tk.Button(bar, text="Save (Ctrl+S)", command=self.save).pack(side="right")
        root.bind("<Alt-Left>", lambda e: self.back())
        root.bind("<Alt-Right>", lambda e: self.forward())
        root.bind("<Control-s>", lambda e: self.save())
        for box in (self.upper_box, self.lower_box):
            box.bind("<Control-a>", self.select_all)
            box.bind("<Control-A>", self.select_all)
            box.bind("<Button-1>", lambda e: e.widget.focus_set())
        self.show()

    def select_all(self, event):
        box = event.widget
        box.tag_remove("sel", "1.0", "end")
        box.tag_add("sel", "1.0", "end-1c")
        box.mark_set("insert", "end-1c")
        return "break"

    def store(self):
        self.steps[self.index]["lower"] = self.lower_box.get("1.0", "end").strip()

    def show(self):
        step = self.steps[self.index]
        self.status.config(text="Sentence step %d / %d" % (self.index + 1, len(self.steps)))
        self.upper_box.config(state="normal")
        self.upper_box.delete("1.0", "end")
        self.upper_box.insert("1.0", step["upper"] if step["had_upper"] else "(no counterpart in the upper segment)")
        self.upper_box.config(state="disabled")
        self.lower_box.delete("1.0", "end")
        self.lower_box.insert("1.0", step["lower"])
        self.lower_box.edit_reset()
        if not step["had_lower"]:
            self.note.config(text="Nothing in the lower segment for this step. Type a sentence to add one, or leave empty.")
        elif not step["had_upper"]:
            self.note.config(text="This lower sentence has no counterpart in the upper segment.")
        else:
            self.note.config(text="")
        self.lower_box.focus_set()

    def back(self):
        self.store()
        if self.index > 0:
            self.index -= 1
            self.show()

    def forward(self):
        self.store()
        if self.index < len(self.steps) - 1:
            self.index += 1
            self.show()

    def result(self):
        parts = [s["lower"] for s in self.steps if s["lower"].strip()]
        return " ".join(parts)

    def save(self):
        self.store()
        text = self.result()
        with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.config(text="Saved to %s and copied to clipboard (step %d / %d)" % (OUTPUT_FILE, self.index + 1, len(self.steps)))

def main():
    try:
        steps = build_steps()
    except (OSError, ValueError) as error:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", str(error))
        return
    if not steps:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", "No sentences found.")
        return
    root = tk.Tk()
    Editor(root, steps)
    root.mainloop()

if __name__ == "__main__":
    main()
