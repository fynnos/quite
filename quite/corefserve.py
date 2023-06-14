import requests
from spacy.language import Language
from spacy.tokens import Doc, Span, Token


@Language.factory(
    "corefserve",
    default_config={
        "url": "http://localhost:1080/predictions/tuba10_electra_uncased_512_3"
    },
)
def statementFactory(nlp, name, url):
    return CorefServe(url)


class CorefServe(object):
    """Coreference resolution using TorchServe"""

    name = "coref_serve"  # component name, will show up in the pipeline

    def __init__(
        self, url="http://localhost:1080/predictions/tuba10_electra_uncased_512_3"
    ):
        self.url = url
        Token.set_extension("is_coref", default=False)
        Token.set_extension("corefs", default=[])
        Token.set_extension("mention_ids", default=[])
        Span.set_extension("is_coref", getter=self.__is_coref)
        Span.set_extension("corefs", getter=self._corefs)
        Doc.set_extension("coref_cluster", default=[])

    def __is_coref(self, tokens):
        return all(map(lambda t: len(t._.corefs), tokens))

    def _corefs(self, tokens):
        if len(tokens) == 0:
            return []
        elif len(tokens) == 1:
            return tokens[0]._.corefs
        result = tokens[0]._.corefs
        for t in tokens:
            for ref in t._.corefs:
                if ref not in result:
                    result.append(ref)
        return result

    def __call__(self, doc: Doc):
        tokens = [[t.text for t in s] for s in doc.sents]
        data = {"output_format": "list", "tokenized_sentences": tokens}
        r = requests.post(self.url, json=data)
        if r.status_code != 200:
            raise Exception("CoRef server failur")
        pred = r.json()
        clusters = []
        for i, cluster in enumerate(pred):
            c = Cluster(i)
            for m, mention in enumerate(cluster):
                c.members.append(doc[mention[0] : mention[1] + 1])
                for t in doc[mention[0] : mention[1] + 1]:
                    t._.corefs.append(c)
                    t._.mention_ids.append(m)
            c.main = c.members[0]
            c.main_ = c.main.text
            ents = c.main.ents
            if len(ents) > 0:
                c.label = ents[0].label_
            clusters.append(c)
        doc._.set("coref_cluster", clusters)
        return doc


class Cluster(object):
    """A cluster consists of usually two or more mentions referring to the same entity"""

    def __init__(self, id):
        self.main = None
        self.main_ = None
        self.members = []
        self.id = id
        self.i = self.id

    def __repr__(self):
        return f"{self.id}: {self.main_} ({self.label}): {self.members}"
