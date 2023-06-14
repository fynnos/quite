from iwnlp.iwnlp_wrapper import IWNLPWrapper
from spacy.language import Language
from spacy.tokens import Token, Doc


@Language.factory(
    "iwnlp", default_config={"datapath": "data/IWNLP.Lemmatizer_20181001.json"}
)
def statementFactory(nlp, name, datapath):
    return IwnlpSpacy(datapath)


class IwnlpSpacy(object):
    def __init__(self, datapath="data/IWNLP.Lemmatizer_20181001.json"):
        Token.set_extension("iwnlp_lemmas", getter=self._get_lemma)
        self.lem = IWNLPWrapper(lemmatizer_path=datapath)

    def __call__(self, doc):
        return doc

    def _get_lemma(self, tok: Token):
        return self.lem.lemmatize(tok.text, tok.pos_)

    def lemmatize(self, word: str, pos: str):
        return self.lem.lemmatize(word, pos)
