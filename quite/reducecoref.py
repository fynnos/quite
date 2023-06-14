from spacy.tokens import Doc, Span, Token
from spacy.language import Language
from quite.statement import Statement
from quite.corefserve import Cluster


@Language.factory("reducecoref")
def statementFactory(nlp, name):
    return ReduceCoref()


class ReduceCoref(object):
    """Removes coreferences not participating in any statements"""

    name = "reduce_coref"  # component name, will show up in the pipeline

    def __call__(self, doc: Doc):
        clusters = doc._.coref_cluster
        for cluster in clusters:
            keep = False
            for mention in cluster.mentions:
                if mention._.has_stm:
                    keep = True
                    break
            if not keep:
                cluster.mentions.clear()
        return doc
