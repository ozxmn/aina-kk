LOW = {"а":"a","ә":"ä","б":"b","в":"v","г":"g","ғ":"ğ","д":"d","е":"e","ё":"io","ж":"j","з":"z","и":"i","й":"i","к":"k","қ":"q","л":"l",
       "м":"m","н":"n","ң":"ñ","о":"o","ө":"ö","п":"p","р":"r","с":"s","т":"t","у":"u","ұ":"ū","ү":"ü","ф":"f","х":"h","һ":"h","ц":"ts",
       "ч":"tş","ш":"ş","щ":"ştş","ъ":"","ы":"y","і":"ı","ь":"","э":"e","ю":"iu","я":"ia"}
UP = {"И":"İ","Й":"İ","І":"I","Ё":"İo","Ю":"İu","Я":"İa","Ч":"Tş","Щ":"Ştş","Ц":"Ts"}
def to_latin(t):
    out = []
    for c in t:
        if c in UP: out.append(UP[c])
        elif c.lower() in LOW:
            r = LOW[c.lower()]
            out.append(r.upper() if (c.isupper() and len(r) == 1) else r)
        else: out.append(c)
    return "".join(out)

VOW = set("aäeioöuūüyıAÄEIOÖUŪÜYIİ")
INV1 = {"a":"а","ä":"ә","b":"б","v":"в","g":"г","ğ":"ғ","d":"д","e":"е","j":"ж","z":"з","k":"к","q":"қ","l":"л","m":"м","n":"н","ñ":"ң",
        "o":"о","ö":"ө","p":"п","r":"р","s":"с","t":"т","u":"у","ū":"ұ","ü":"ү","f":"ф","h":"х","ş":"ш","y":"ы","ı":"і","c":"ц","x":"х","w":"у"}
def to_cyrillic(t):
    out, i, n = [], 0, len(t)
    while i < n:
        c = t[i]; lo = c.lower(); up = c.isupper() or c == "İ"
        prev = t[i - 1] if i else " "; nxt = t[i + 1] if i + 1 < n else " "
        def emit(cyr, k):
            out.append(cyr.upper() if up and len(cyr) == 1 else (cyr[0].upper() + cyr[1:] if up else cyr)); return k
        if t[i:i + 3].lower() == "ştş": i += emit("щ", 3); continue
        if t[i:i + 2].lower() == "tş": i += emit("ч", 2); continue
        if t[i:i + 2].lower() == "ts" and not prev.isalpha(): i += emit("ц", 2); continue
        if c == "I": out.append("І"); i += 1; continue
        if lo in ("i", "i̇") or c == "İ":
            if nxt.lower() in ("a", "u", "o") and (not prev.isalpha() or prev in VOW):
                i += emit({"a": "я", "u": "ю", "o": "ё"}[nxt.lower()], 2); continue
            i += emit("й" if prev in VOW else "и", 1); continue
        if c == "I": out.append("І"); i += 1; continue
        if lo in INV1: i += emit(INV1[lo], 1); continue
        out.append(c); i += 1
    return "".join(out)

if __name__ == "__main__":
    print(to_latin("Қазақстан Республикасы — Орталық Азиядағы мемлекет. Ілияс Жансүгіров, Шыңғыс, Чапаев, Іле, Ищанов, Юрий."))
