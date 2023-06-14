import requests
from spacy.tokens import Doc, Span, Token


class MicroNer(object):
    """MicroNer Client"""

    name = "micro_ner"  # component name, will show up in the pipeline

    def __init__(self, nlp, url="http://localhost:5001/api"):
        """
        Initialise the pipeline component. The shared nlp instance is used
        to initialise the matcher with the shared vocab, get the label ID and
        generate Doc objects as phrase match patterns.
        """
        self.strings = nlp.vocab.strings  # get entity label ID
        self.mapper = {}
        self.url = url
        Token.set_extension("ent_iob", default=0)
        Doc.set_extension("is_nered", default=False)

    def _label(self, label):
        id = self.mapper.get(label, None)
        if id is not None:
            return id
        id = self.strings[label]
        self.mapper[label] = id
        return id

    def __call__(self, doc: Doc):
        """
        Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """

        tokens = [[t.text for t in s] for s in doc.sents]
        # MicroNer only supports 100 tokens per sentence max
        tokens = [s if len(s) <= 100 else s[:100] for s in tokens]
        data = {
            "meta": {"model": "germeval-conll.h5"},
            "data": {"sentences": [], "tokens": tokens},
        }
        r = requests.post(self.url, json=data)
        bio = r.json()["tokens"]

        ents = []
        label = None
        label_id = 0
        start = 0
        end = 0
        for tokens, annotations in zip(doc.sents, bio):
            for t, a in zip(tokens, annotations):
                tag = a[1]
                if tag[0] == "B":
                    start = t.i
                    end = start
                    label = tag[2:]
                    label_id = self._label(label)
                    # t.ent_iob = 3
                    t._.set("ent_iob", 3)
                    t.ent_type = label_id
                elif tag[0] == "I" and label == tag[2:]:
                    end = t.i
                    t.ent_type = label_id
                    # t.ent_iob = 1
                    t._.set("ent_iob", 1)
                elif label is not None:
                    entity = Span(doc, start, end + 1, label=label_id)
                    ents.append(entity)
                    label = None
                    # t.ent_iob = 2
                    t._.set("ent_iob", 2)
                else:
                    t._.set("ent_iob", 2)
        if label is not None:
            entity = Span(doc, start, end + 1, label=label_id)
            ents.append(entity)
            label = None
            # t.ent_iob = 2
            t._.set("ent_iob", 2)
        doc.ents = ents
        doc._.set("is_nered", True)
        return doc
