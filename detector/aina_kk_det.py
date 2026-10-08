import json, os, re, sys, unicodedata
import numpy as np

OBSERVER = "google/gemma-4-E2B"
PERFORMER = "google/gemma-4-E2B-it"
SEEDS = ("seed1", "seed2", "seed3")
MAX_LEN = 512

LAT2CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і", "A": "А", "E": "Е", "O": "О", "P": "Р",
           "C": "С", "T": "Т", "H": "Н", "K": "К", "M": "М", "B": "В", "X": "Х", "I": "І"}
VOW = set("aäeioöuūüyıAÄEIOÖUŪÜYIİ")
INV1 = {"a": "а", "ä": "ә", "b": "б", "v": "в", "g": "г", "ğ": "ғ", "d": "д", "e": "е", "j": "ж", "z": "з", "k": "к", "q": "қ", "l": "л",
        "m": "м", "n": "н", "ñ": "ң", "o": "о", "ö": "ө", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у", "ū": "ұ", "ü": "ү", "f": "ф",
        "h": "х", "ş": "ш", "y": "ы", "ı": "і", "c": "ц", "x": "х", "w": "у"}
MD = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)", re.M)
MD_HEADER = re.compile(r"(?m)^\s{0,3}#{1,6}\s[^\n]*\n?")


def strip_markup(t):
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", t)
    t = MD.sub("", t)
    t = re.sub(r"`+", "", t)
    return t


def normalize(t):
    t = unicodedata.normalize("NFC", t)
    t = strip_markup(t)
    t = re.sub("[­​‌‍⁠﻿]", "", t)
    t = re.sub("[  -   　]", " ", t)
    t = re.sub(r"[“”„]", "\"", t)
    t = re.sub(r"[‘’‚]", "'", t)
    t = re.sub(r"[«»]", "\"", t)
    t = re.sub("[‐-―−]", "-", t)
    t = t.replace("…", "...")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def strip_title(t, max_words=12, min_rest=40):
    lines = t.strip().split("\n")
    if len(lines) < 2:
        return t
    first = re.sub(r"\*\*|__", "", lines[0]).strip()
    rest = "\n".join(lines[1:]).strip()
    if first and len(first.split()) <= max_words and not re.search(r"[.!?…:;,]\s*$", first) and len(rest.split()) >= min_rest:
        return rest
    return t


def fold_homoglyphs(t):
    def fix(m):
        w = m.group(0)
        cyr = sum("Ѐ" <= c <= "ӿ" for c in w)
        if cyr == 0 or cyr < len(w) / 2:
            return w
        return "".join(LAT2CYR.get(c, c) for c in w)
    return re.sub(r"\w+", fix, t)


def to_cyrillic(t):
    out, i, n = [], 0, len(t)
    while i < n:
        c = t[i]
        lo = c.lower()
        up = c.isupper() or c == "İ"
        prev = t[i - 1] if i else " "
        nxt = t[i + 1] if i + 1 < n else " "

        def emit(cyr, k):
            out.append(cyr.upper() if up and len(cyr) == 1 else (cyr[0].upper() + cyr[1:] if up else cyr))
            return k
        if t[i:i + 3].lower() == "ştş":
            i += emit("щ", 3)
            continue
        if t[i:i + 2].lower() == "tş":
            i += emit("ч", 2)
            continue
        if t[i:i + 2].lower() == "ts" and not prev.isalpha():
            i += emit("ц", 2)
            continue
        if c == "I":
            out.append("І")
            i += 1
            continue
        if lo in ("i", "i̇") or c == "İ":
            if nxt.lower() in ("a", "u", "o") and (not prev.isalpha() or prev in VOW):
                i += emit({"a": "я", "u": "ю", "o": "ё"}[nxt.lower()], 2)
                continue
            i += emit("й" if prev in VOW else "и", 1)
            continue
        if lo in INV1:
            i += emit(INV1[lo], 1)
            continue
        out.append(c)
        i += 1
    return "".join(out)


def script_of(t):
    cyr = sum("Ѐ" <= c <= "ӿ" for c in t)
    lat = sum(c.isalpha() and c < "ɐ" for c in t)
    return "latin" if lat > cyr else "cyrillic"


def prepare(text):
    t = MD_HEADER.sub("", text)
    t = strip_title(t)
    script = script_of(t)
    if script == "latin":
        t = to_cyrillic(t)
    t = fold_homoglyphs(t)
    return normalize(t), script


def _token_stats(model, tok, texts, device):
    import torch
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN, add_special_tokens=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[:, :-1]
    return logits, enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:].float()


def _binoculars(obs_logits, perf_logits, labels, mask, chunk=128):
    import torch
    import torch.nn.functional as F
    dev = perf_logits.device
    B, T = labels.shape
    s_ppl = torch.zeros(B, device=dev)
    s_x = torch.zeros(B, device=dev)
    with torch.no_grad():
        for a in range(0, T, chunk):
            plp = F.log_softmax(perf_logits[:, a:a + chunk].float(), -1)
            op = F.softmax(obs_logits[:, a:a + chunk].to(dev).float(), -1)
            lab = labels[:, a:a + chunk].to(dev)
            m = mask[:, a:a + chunk].to(dev)
            s_ppl += (-(plp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)) * m).sum(1)
            s_x += (-(op * plp).sum(-1) * m).sum(1)
            del plp, op
    n = mask.to(dev).sum(1).clamp(min=1)
    return (s_ppl / n) / (s_x / n)


class AinaKKDet:
    def __init__(self, path=".", classifier_device=None, observer_device=None, performer_device=None):
        import torch
        self.path = path
        cp = os.path.join(path, "calibration.json")
        self.cal = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else None
        gpu = torch.cuda.is_available()
        n_gpu = torch.cuda.device_count() if gpu else 0
        self.classifier_device = classifier_device or ("cuda:0" if gpu else "cpu")
        self.observer_device = observer_device or ("cuda:0" if gpu else "cpu")
        self.performer_device = performer_device or ("cuda:1" if n_gpu > 1 else self.observer_device)
        self._clf = None
        self._lm = None
        if self.cal:
            self._ref_c = np.sort(np.array(self.cal["dev_human_classifier"], dtype=float))
            self._ref_b = np.sort(-np.array(self.cal["dev_human_binoculars"], dtype=float))

    @classmethod
    def from_pretrained(cls, repo_id="ajnasiou/aina-kk-det", **kw):
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo_id), **kw)

    def _load_classifier(self):
        if self._clf is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._clf = []
            for s in SEEDS:
                d = os.path.join(self.path, s)
                tok = AutoTokenizer.from_pretrained(d)
                m = AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float32).to(self.classifier_device).eval()
                self._clf.append((tok, m))
        return self._clf

    def _load_lm(self):
        if self._lm is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            dt = torch.float16 if self.observer_device.startswith("cuda") else torch.float32
            tok = AutoTokenizer.from_pretrained(OBSERVER)
            tok.padding_side = "right"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            obs = AutoModelForCausalLM.from_pretrained(OBSERVER, dtype=dt).to(self.observer_device).eval()
            perf = AutoModelForCausalLM.from_pretrained(PERFORMER, dtype=dt).to(self.performer_device).eval()
            self._lm = (tok, obs, perf)
        return self._lm

    def unload(self):
        import gc
        import torch
        self._clf = None
        self._lm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classifier_scores(self, texts, batch_size=32):
        import torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        total = np.zeros(len(texts))
        for tok, m in self._load_classifier():
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(self.classifier_device)
                with torch.no_grad():
                    p = torch.softmax(m(**enc).logits.float(), -1)[:, 1]
                total[idx] += p.cpu().numpy()
        return (total / len(SEEDS)).tolist()

    def binoculars_scores(self, texts):
        tok, obs, perf = self._load_lm()
        out = []
        for t in texts:
            lo, _, _ = _token_stats(obs, tok, [t], self.observer_device)
            lp, lab, mk = _token_stats(perf, tok, [t], self.performer_device)
            out.append(float(_binoculars(lo, lp, lab, mk)[0]))
            del lo, lp
        return out

    def _ecdf(self, ref, x):
        return float(np.searchsorted(ref, x, side="right")) / len(ref)

    def bucket(self, n_words):
        for name, lo in self.cal["length_buckets"]:
            if n_words >= lo:
                return name
        return None

    def decide(self, text, classifier, binoculars):
        n = len(text.split())
        b = self.bucket(n)
        combined = max(self._ecdf(self._ref_c, classifier), self._ecdf(self._ref_b, -binoculars))
        r = {"n_words": n, "length_bucket": b, "classifier": classifier, "binoculars": binoculars, "combined": combined}
        if b is None:
            r.update({"machine": None, "threshold": None,
                      "note": f"text shorter than {self.cal['length_buckets'][-1][1]} words; no calibrated threshold"})
            return r
        th = self.cal["thresholds"][b]
        r.update({"machine": combined >= th, "threshold": th, "test_rates": self.cal["test_rates"].get(b)})
        return r

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        prepared = [prepare(t) for t in texts]
        clean = [p[0] for p in prepared]
        c = self.classifier_scores(clean)
        b = self.binoculars_scores(clean)
        out = []
        for (t, script), ci, bi in zip(prepared, c, b):
            r = self.decide(t, ci, bi)
            r["script"] = script
            out.append(r)
        return out


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="aina_kk_det")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--model", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args(argv)
    texts = [open(f, encoding="utf-8").read() for f in a.files] if a.files else [sys.stdin.read()]
    det = AinaKKDet(a.model)
    for name, r in zip(a.files or ["<stdin>"], det.predict(texts)):
        r["file"] = name
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
