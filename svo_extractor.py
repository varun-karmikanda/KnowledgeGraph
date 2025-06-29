import spacy, json, os
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 4_000_000


def extract_text(path):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p"))

def extract_svos(doc):
    svos = []
    for i, sent in enumerate(doc.sents):
        subj, verb, obj = "", "", ""
        for token in sent:
            if "subj" in token.dep_: subj = token.text
            if token.pos_ == "VERB": verb = token.lemma_
            if "obj" in token.dep_: obj = token.text
        if subj and verb and obj:
            svos.append({"subject": subj, "verb": verb, "object": obj, "sentence_id": i, "sentence": sent.text})
    return svos

text = extract_text("data/raw_text/hello.html")
doc = nlp(text)
os.makedirs("data", exist_ok=True)
with open("data/svo_triplets.json", "w") as f:
    json.dump(extract_svos(doc), f, indent=2)
