from dawg import DAWG, RecordDAWG
from demorphy import Analyzer
from demorphy.cache import lrudecorator


class Morphology(object):
    def __init__(self):
        az = Analyzer()
        cached = lrudecorator(256)
        self.analyze = cached(az.analyze)

    def tense(self, word: str):
        result = self.analyze(word)
        ret = set()
        for r in result:
            t = r.tense
            if t:
                ret.add(t)
        return tuple(ret)

    def mode(self, word: str):
        result = self.analyze(word)
        return self._mode(result)

    def _mode(self, analysis):
        ret = set()
        for a in analysis:
            m = a.mode
            if m:
                ret.add(m)
        return tuple(ret)

    def is_subjunctive(self, verb: str, subject: str):
        v = list(filter(lambda a: a.mode, self.analyze(verb)))
        vm = self._mode(v)
        if len(vm) == 1 and vm[0] == "subj":
            return True
        if "subj" not in vm:
            return False

        if subject and subject.lower() not in ("ich", "du", "wir", "ihr"):
            s = self.analyze(subject)
            if not s:
                s = self.analyze(subject.lower())
            num = []
            for i in s:
                if i.case == "nom":
                    num.append(i.numerus)
            matches = 0
            for i in v:
                if i.person == "3per" and i.numerus in num:
                    if i.mode == "subj":
                        matches += 1
                    else:
                        return False
            return matches > 0

        return False
