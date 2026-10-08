import re, unicodedata
KZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
LAT2CYR = {"a":"а","e":"е","o":"о","p":"р","c":"с","y":"у","x":"х","i":"і","A":"А","E":"Е","O":"О","P":"Р","C":"С","T":"Т","H":"Н","K":"К","M":"М","B":"В","X":"Х","I":"І"}
CYR2LAT = {"а":"a","е":"e","о":"o","р":"p","с":"c","у":"y","х":"x","і":"i","А":"A","Е":"E","О":"O","Р":"P","С":"C","Т":"T","Н":"H","К":"K","М":"M","В":"B","Х":"X","І":"I"}
MD = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)", re.M)

def strip_markup(t):
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", t)
    t = MD.sub("", t)
    t = re.sub(r"`+", "", t)
    return t

def normalize(t):
    t = unicodedata.normalize("NFC", t)
    t = strip_markup(t)
    t = re.sub("[\u00ad\u200b\u200c\u200d\u2060\ufeff]", "", t)
    t = re.sub("[\u00a0\u2000-\u200a\u202f\u205f\u3000]", " ", t)
    t = re.sub(r"[“”„]", "\"", t); t = re.sub(r"[‘’‚]", "'", t)
    t = re.sub(r"[«»]", "\"", t)
    t = re.sub("[\u2010-\u2015\u2212]", "-", t)
    t = t.replace("…", "...")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()

MD_HEADER = re.compile(r"(?m)^\s{0,3}#{1,6}\s[^\n]*\n?")
def drop_md_headers(t):
    return MD_HEADER.sub("", t)

def strip_title(t, max_words=12, min_rest=40):
    lines = t.strip().split("\n")
    if len(lines) < 2: return t, False
    first = re.sub(r"\*\*|__", "", lines[0]).strip()
    rest = "\n".join(lines[1:]).strip()
    if first and len(first.split()) <= max_words and not re.search(r"[.!?…:;,]\s*$", first) and len(rest.split()) >= min_rest:
        return rest, True
    return t, False

def to_sentence_end(t):
    m = list(re.finditer(r"[.!?…»\"”)](?=\s|$)", t))
    return t[: m[-1].end()] if m else t

ODD = "Ͱ-Ͽ԰-֏֐-׿؀-ۿऀ-ॿႠ-ჿ぀-ヿ一-鿿가-힯"
MIXED_TOKEN = re.compile(rf"[Ѐ-ӿ][{ODD}]|[{ODD}][Ѐ-ӿ]")
def mixed_script_token(t):
    return bool(MIXED_TOKEN.search(t))

def fold_homoglyphs(t):
    def fix(m):
        w = m.group(0)
        cyr = sum('Ѐ' <= c <= 'ӿ' for c in w)
        if cyr == 0 or cyr < len(w) / 2: return w
        return "".join(LAT2CYR.get(c, c) for c in w)
    return re.sub(r"\w+", fix, t)

def homoglyph_count(t):
    n = 0
    for w in re.findall(r"\w+", t):
        cyr = sum('Ѐ' <= c <= 'ӿ' for c in w)
        if cyr and cyr >= len(w) / 2: n += sum(c in LAT2CYR for c in w)
    return n

def kz_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(c in KZ_LETTERS for c in letters) / max(1, len(letters))

PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
       (re.compile(r"(?<!\d)\d{12}(?!\d)"), "[IIN]"),
       (re.compile(r"(?:\+7|8)[\s(-]*7\d{2}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}"), "[PHONE]")]
def scrub_pii(t):
    for rx, rep in PII: t = rx.sub(rep, t)
    return t
