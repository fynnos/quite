from spacy.tokens import Doc, Span, Token
from spacy.pipeline.lemmatizer import Lemmatizer
from spacy.language import Language
from typing import NamedTuple, Tuple
from sys import stderr, maxsize
import regex as re
import os
from quite.morph import Morphology
from quite.util import span_overlap, span_union
from quite.iwnlpspacy import IwnlpSpacy
from quite.human import is_human

_ext_stmpart = "stmpart"
_ext_statements = "statements"
_ext_is_quote = "is_quote"
_ext_stm = "stm"

_quote_regex_str = r'(?:^|(?<=\W))(?:([\'"])|(“)|(»)|(«)|(‚)|(‘)|(›)|(‹)|(„)).+?(?:(?(2)[”“]|\0)|(?(1)\1|\0)|(?(3)«|\0)|(?(4)»|\0)|(?(5)‘|\0)|(?(5)‘|\0)|(?(6)’|\0)|(?(7)‹|\0)|(?(8)›|\0)|(?(9)“|\0))(?=\W|$)'
_quote_regex = re.compile(_quote_regex_str)

_pos_verbs = ("VERB", "AUX")


class Statement(NamedTuple):
    id: int
    subject: Span
    cue: Token
    cite: Span
    origin: str

    def _tag_tokens(self):
        st = self
        for t in st.cite:
            t._.stmpart = 3
            t._.stm.append(st)
        if st.subject:
            try:
                for t in st.subject:
                    t._.stmpart = 1
                    if st not in t._.stm:
                        t._.stm.append(st)
            except AssertionError:
                st.subject._.stmpart = 1
                if st not in st.subject._.stdm:
                    st.subject._.stdm.append(st)
        if st.cue:
            st.cue._.stmpart = 2
            if st not in st.cue._.stm:
                st.cue._.stm.append(st)
        return self


@Language.factory(
    "statements",
    default_config={"language": "de", "use_iwnlp": False, "use_morph": False},
)
def statementFactory(nlp, name, language, use_iwnlp, use_morph):
    return Statements(nlp, language, use_iwnlp, use_morph)


class Statements(object):
    """German statement, (in)direct speech finder"""

    name = "Statements"  # component name, will show up in the pipeline
    id_to_tag = [None, "SUBJ", "CUE", "CITE"]

    def __init__(self, model, language="de", use_iwnlp=False, use_morph=False):
        Token.set_extension(_ext_stmpart, default=0)
        Token.set_extension(_ext_stm, default=[])
        Token.set_extension(_ext_is_quote, default=False)
        Span.set_extension("has_stm", getter=self._has_stm)
        Span.set_extension(_ext_statements, getter=self._statements)
        Doc.set_extension(_ext_statements, default=[])

        self._use_iwnlp = use_iwnlp
        self._use_morph = use_morph
        self._dp_subjects = (
            ("sb", "sbp", "sp") if language == "de" else ("nsubj", "nsubjpass")
        )
        self._dp_subject_grows = (
            ("nk", "pnc", "pg", "cj", "cd")
            if language == "de"
            else ("compound", "nn", "amod", "nounmod")
        )
        self._dp_contents = (
            ("oc", "op", "re") if language == "de" else ("ccomp",)
        )  # oc: (in)directe rede. op,re,mo: reported speech/thought
        self._dp_objects = ("oa",) if language == "de" else ()
        self._dp_reasons = ("mo", "nk") if language == "de" else ()
        self._forbidden_subjects = (
            (
                "es",
                "man",
                "wer",
                "dies",
                "jenes",
                "dieses",
                "niemand",
                "jeder",
                "keiner",
                "das",
                "einer",
                "jemand",
            )
            if language == "de"
            else ("who",)
        )
        self._lang = language
        self._lem = (
            [p for p in model.pipeline if p[0] == "iwnlp"][0][1] if use_iwnlp else None
        )

        with open(
            os.path.join(os.path.dirname(__file__), f"data/verbs-{language}.txt"), "r"
        ) as file:
            self._cues = frozenset(file.read().splitlines())
        if use_morph:
            self._morph = Morphology()

    def __call__(self, doc: Doc):
        if not doc.is_nered and not doc._.is_nered:
            raise NotNeredException

        di = self._directs(doc, self._lem)
        indi = self._indirect(doc, self._lem)
        statements = [
            Statement(i, subj, cue, cite, origin)._tag_tokens()
            for i, (subj, cue, cite, origin) in enumerate(self._fuse2(di, indi))
        ]
        doc._.statements = statements
        return doc

    def _fuse2(self, di, indi):
        # TODO fusing needs work: often duplicate indirect parts (at the end) of longer direct citation
        s = object()
        d = next(di, s)
        i = next(indi, s)
        while d is not s and i is not s:
            if len(d) == 2 and len(i) == 2:
                a1, a2 = i
                b1, b2 = d
                if a1 <= b1 and a2 <= b1:
                    i = next(indi, s)
                elif a1 >= b2 and a2 >= b2:
                    d = next(di, s)
                else:
                    # there is some potential overlap
                    overlap = min(a2, b2) - max(a1, b1)
                    assert overlap > 0, "There should be an overlap but its not?!"
                    # if b1 >= a1 and b2 <= a2:
                    #     # d in i -> take only i
                    #     i = next(indi, s)
                    # elif a1 >= b1 and a2 <= b2:
                    #     # i in d -> take only d
                    #     d = next(di, s)
                    if overlap * 2 >= a2 - a1 or overlap * 2 >= b2 - b1:
                        # more than half of the smaller span is overlapped -> take both
                        if a1 <= b1:
                            i = next(indi, s)
                            d = next(di, s)
                        else:
                            d = next(di, s)
                            i = next(indi, s)
                    else:
                        # simply start with the first (or indirect if equal)
                        if a1 <= b1:
                            i = next(indi, s)
                        else:
                            d = next(di, s)
            elif len(d) == 3 and len(i) == 3:
                a1, a2 = i[2].start, i[2].end
                b1, b2 = d[2].start, d[2].end
                overlap = min(a2, b2) - max(a1, b1)
                if 2 * overlap > min(a2 - a1, b2 - b1):  #  and d[0] == i[0]:
                    # TODO if i is fully contained in d, discard i
                    if a1 > b1 and a2 < b2:
                        # i fully contained in d, discard i
                        yield *d, "discard i"
                    elif a1 < b1 and a2 > b2:
                        # d fully contained in i, discard d
                        yield *i, "discard d"
                    else:
                        # combine
                        cite = i[2].doc[
                            min(i[2].start, d[2].start) : max(i[2].end, d[2].end)
                        ]
                        if d[0] != i[0]:
                            print("missed no over", d, file=stderr)
                        yield i[0], i[1], cite, "no over"
                else:
                    # overlap too small #(or subject not identical)
                    yield *i, "overlap i"
                    yield *d, "overlap d"
                i = next(indi, s)
                d = next(di, s)

            elif len(d) == 3:
                yield *d, "both d"
                d = next(di, s)
            elif len(i) == 3:
                if not i[2]._.has_stm and not any(
                    self.is_stm_contained(i, s) for s in i[2]._.statements
                ):
                    yield *i, "both i"
                else:
                    print("missed both i", i, file=stderr)
                i = next(indi, s)
        while d is not s:
            if len(d) == 3:
                yield *d, "only d"
            d = next(di, s)
        while i is not s:
            if len(i) == 3:
                if not i[2]._.has_stm and not any(
                    self.is_stm_contained(i, s) for s in i[2]._.statements
                ):
                    yield *i, "only i"
                else:
                    print("missed only i", i, file=stderr)
            i = next(indi, s)

    @staticmethod
    def is_stm_contained(stm1: Tuple, stm2: Statement):
        c1 = stm1[2]
        c2 = stm2.cite
        return c1.start >= c2.start and c1.end <= c2.end

    def _directs(self, doc, lem):
        for q, certain in self._quotes(doc):
            yield q.start, q.end
            match = self._direct2(q, certain, lem, doc)
            if match:
                yield match

    def _direct2(self, q: Span, certain: bool, lem: Lemmatizer, doc: Doc):
        start: Span = q[1].sent
        end: Span = q[-2].sent

        is_single_sentence: bool = start == end
        is_open_start: bool = not self._contains(q, start)
        is_open_end: bool = not self._contains(q, end)
        has_front_indicator: bool = self._quote_front_indicator(q)
        has_back_indicator: bool = self._quote_back_indicator(q)

        try:
            before: Span = (
                doc[start.start - 2].sent if is_open_start else doc[q.start - 2].sent
            )
        except IndexError:
            before: Span = start
        try:
            after: Span = doc[end.end + 1].sent if is_open_end else doc[q.end + 1].sent
        except IndexError:
            after: Span = end

        has_prev_stm: bool = self._has_stm(before)
        has_cur_stm: bool = self._has_stm(start)

        actions = []
        if is_single_sentence:
            if is_open_start:
                actions.append((1,))
                actions.append((5,))
        else:
            if is_open_start & is_open_end:
                actions.append((1, 2))
                actions.append((5,))
            elif is_open_start:
                actions.append((1,))
                actions.append((5,))
            elif is_open_end:
                actions.append((2,))

        if has_front_indicator & has_back_indicator:
            actions.append((0, 3))
            actions.append((4,))
        elif has_front_indicator:
            actions.append((0, 4))
            actions.append((3,))
        elif has_back_indicator:
            actions.append((3,))
            actions.append((0, 4))
        else:
            actions.append((4,))
            actions.append((0, 3))

        cue_before, subj_before = self._outer_cue_subj(doc, lem, before, q)
        cue_start, subj_start = self._outer_cue_subj(doc, lem, start, q)
        cue_end, subj_end = self._outer_cue_subj(doc, lem, end, q)
        cue_after, subj_after = self._outer_cue_subj(doc, lem, after, q)
        cue_prev, subj_prev = (
            None,
            self._statements(before)[-1].subject if has_prev_stm else None,
        )
        cue_cur, subj_cur = (
            None,
            self._statements(start)[0].subject if has_cur_stm else None,
        )

        cues = [cue_before, cue_start, cue_end, cue_after, cue_prev, cue_cur]
        subjects = [subj_before, subj_start, subj_end, subj_after, subj_prev, subj_cur]

        return self._decide_action(actions, subjects, cues, q)

    def _decide_action(self, actions: list, subjects: list, cues: list, quote: Span):
        for action in actions:
            candidates = [subjects[a] for a in action]
            nearest = self._nearest(quote, candidates)
            if nearest is not None:
                tmp_cues = [cues[a] for a in action]
                return candidates[nearest], tmp_cues[nearest], quote
        return None, None, quote

    def _nearest(self, target: Span, candidates: list):
        minimum = maxsize
        nearest = None
        for i, span in enumerate(candidates):
            if span:
                distance: int = min(
                    abs(target.start - span.end), abs(span.start - target.end)
                )
                if distance < minimum:
                    minimum, nearest = distance, i
        return nearest

    def _outer_cue_subj(self, doc: Doc, lem: Lemmatizer, sent: Span, forbidden: Span):
        cue: Token = self._cue(lem, doc, sent.root)
        if cue and (cue.i >= forbidden.start and cue.i < forbidden.end):
            cue = None
        subj: Span = self._subject(doc, cue if cue else sent.root, forbidden)
        # TODO implement these checks for english
        if (
            subj
            and span_overlap(subj, forbidden) == 0
            and (
                self._lang != "de"
                or len(subj.ents) > 0
                or any(t.pos_ in ("PRON", "PROPN") for t in subj)
                or any(is_human(t.text) for t in subj)
            )
        ):
            return cue if cue else None, subj
        return None, None

    def _quotes(self, doc, part=None):
        part = part if part else doc
        for match in _quote_regex.finditer(part.text, overlapped=True):
            # differentiate between opening/closing and universal quotation marks makes no sense
            span = part.char_span(*match.span())
            if span is not None and len(span) > 4:
                before = doc[span.start - 1]
                after = doc[span.end] if span.end < len(doc) else doc[span.end - 1]
                start = span[1]
                end = span[-2]
                marker = span[0]
                endMarker = span[-1]
                markerTuple = (marker.text, endMarker.text)
                text = span[1:-1]
                indicators = 0
                indicators += start.sent != end.sent
                indicators += before.sent == start.sent or after.sent == end.sent
                indicators += start.is_punct
                indicators += end.text in (",", ":")
                indicators += len(marker.whitespace_) > 0
                indicators += len(end.whitespace_) > 0
                indicators += marker._.is_quote
                if indicators > 2:
                    continue
                if start.sent != end.sent:
                    inner = self._quotes(doc, text)
                    innerRes = (
                        q[0].text in markerTuple or q[-1].text in markerTuple
                        for q, c in inner
                    )
                    if len(span) > len(doc) // 4 or any(innerRes):
                        continue
                certain = (
                    before.text == ":"
                    or after.text == ","
                    or end.text in [".", "!", "?", ";"]
                    or self._is_cue(after)
                    or self._is_cue(before)
                )
                if certain:
                    marker._.is_quote = True
                    span[-1]._.is_quote = True
                yield span, certain

    def _quote_front_indicator(self, quote: Span):
        # TODO check up to 3 tokens
        before: Token = quote.doc[quote.start - 1]
        result: bool = (
            before.text in (":", ",")
            or self._is_cue(before)
            or before.ent_type_ in ("PER", "ORG", "PERSON")
        )
        return result

    def _quote_back_indicator(self, quote: Span):
        # TODO check up to 3 tokens
        if len(quote.doc) <= quote.end:
            return False
        after: Token = quote.doc[quote.end]
        result: bool = (
            after.text == ","
            or self._is_cue(after)
            or after.ent_type_ in ("PER", "ORG", "PERSON")
        )
        return result

    def _contains(self, larger: Span, smaller: Span):
        fixed = self._remove_leading_punctuation(smaller)
        fixed = self._remove_trailing_punctuation(fixed)
        return fixed.start >= larger.start and fixed.end <= larger.end

    def _remove_leading_punctuation(self, span: Span):
        i = 0
        while len(span) > i and span[i].is_punct:
            i += 1
        return span[i:]

    def _remove_trailing_punctuation(self, span: Span):
        i = len(span) - 1
        while i > 0 and span[i].is_punct:
            i -= 1
        return span[:i]

    def _indirect(self, doc, lem):
        for sent in doc.sents:
            yield sent.start, sent.end
            s = self._sentence(doc, lem, sent)
            if s is not None and None not in s:
                yield s
            else:
                subject = self._subjunctive(sent, doc)
                if subject:
                    fixed = self._remove_leading_punctuation(sent)
                    yield (subject, None, fixed)

    def _sentence(self, doc, lem, sent):
        verb, subj, cite = self._cue(lem, doc, sent.root), None, None
        if verb is not False and verb is not None:
            obj = self._object(verb)
            cite = self._content(verb)
            if cite is None and obj is not None:
                cite = self._relaxed_content(verb)
            if cite is not None and obj is not None and span_overlap(cite, obj) == 0:
                cite = span_union(obj, cite)
            subj = self._subject(doc, verb, cite)
            if subj is None:
                subj = self._subject(doc, sent.root, cite)
        return (subj, verb, cite)

    def _object(self, verb: Token):
        for c in verb.children:
            if c.dep_ in self._dp_objects:
                l, r = self._grow(c, self._dp_subject_grows, c.i, c.i + 1)
                obj = verb.doc[l:r]
                ne = obj.ents
                return ne[0] if len(ne) == 1 else obj

    def _relaxed_content(self, verb: Token):
        l, r = self._recurse_relaxed_content(verb, verb.right_edge.i, verb.left_edge.i)
        return verb.doc[l : r + 1] if r > l else None

    def _recurse_relaxed_content(self, root: Token, l: int, r: int):
        for c in root.children:
            if c.dep_ in self._dp_contents:
                return self._recurse_content(root, root.right_edge.i, root.left_edge.i)
            if c.dep_ in self._dp_reasons:
                l = min(c.left_edge.i, l)
                r = max(c.right_edge.i, r)
                nl, nr = self._recurse_relaxed_content(c, l, r)
                if nl != nr:
                    return nl, nr
        return 0, 0

    def _content(self, root: Token):
        l, r = self._recurse_content(root, root.right_edge.i, root.left_edge.i)
        return root.doc[l : r + 1] if r > l else None

    def _recurse_content(self, root: Token, l: int, r: int):
        for c in root.children:
            if c.dep_ in self._dp_contents:
                l = min(c.left_edge.i, l)
                r = max(c.right_edge.i, r)
                self._recurse_content(c, l, r)
        return l, r

    def _cue(self, lem, doc, root):
        if root.pos_ in _pos_verbs:
            if self._is_cue(root):
                return root
            for c in root.children:
                if c.dep_ == "svp":  # german only
                    lemma1 = c.text + root.lemma_
                    # TODO replace somehow with _is_cue?
                    if lemma1 in self._cues:
                        return root
                    lemma2 = lem.lemmatize(c.text + root.text, root.pos)
                    if (
                        lemma2 is not None
                        and len(lemma2) > 0
                        and lemma2[0] in self._cues
                    ):
                        return root

            if root.pos_ == "AUX":
                # do not recurse too deep (only for AUX verbs) to avoid false positives
                for c in root.children:
                    ret = self._cue(lem, doc, c)
                    if ret is not None:
                        return ret
        else:
            return None
        return False

    def _subject(self, doc, root, cite=None):
        for c in root.children:
            if c.dep_ in self._dp_subjects:
                l, r = self._grow(c, self._dp_subject_grows, c.i, c.i + 1)
                subj = doc[l:r]
                if len(subj) == 1 and subj[0].lower_ in self._forbidden_subjects:
                    return None
                ne = subj.ents
                return ne[0] if len(ne) == 1 else subj
        ne = (
            root.sent.ents
            if cite is None
            else list(
                filter(
                    lambda n: (n.start < cite.start and n.end <= cite.start)
                    or (n.start >= cite.end and n.end > cite.end),
                    root.sent.ents,
                )
            )
        )
        if len(ne) == 1:
            return ne[0]
        elif len(ne) > 1:
            return self._first(root, ne)
        return None

    def _grow(self, root, dep, l, r):
        for c in root.children:
            if c.dep_ in dep:
                l = min(l, c.i)
                r = max(r, c.i + 1)
                l, r = self._grow(c, dep, l, r)
        return l, r

    def _first(self, root, spans):
        found = None
        for c in root.children:
            for s in spans:
                if c.i >= s.start and c.i <= s.end:
                    return s
            found = self._first(c, spans)
        return found

    def _is_cue(self, tok: Token):
        if self._use_iwnlp and tok._.iwnlp_lemmas is not None:
            for lemma in tok._.iwnlp_lemmas:
                if lemma in self._cues:
                    return True
        return tok.lemma_ in self._cues

    def _subjunctive(self, sent: Span, doc: Doc):
        # check for sentence in Subjunctive Mood (hypothetical) directly after statement
        root = sent.root
        if root.pos_ not in _pos_verbs:
            return False
        subject = None
        for c in root.children:
            if c.dep_ == "sb":
                subject = c.text
                break
        if not self._use_morph:
            return False
        yes = self._morph.is_subjunctive(root.text, subject)
        if yes and (subject is None or subject.lower() not in ("ich", "wir")):
            before = doc[sent.start - 2].sent
            if self._has_stm(before):
                stm = self._statements(before)[-1]
                return stm.subject
        return False

    def _has_stm(self, tokens):
        result = any(map(lambda t: t._.stmpart != 0 and t._.is_quote is False, tokens))
        return result

    def _statements(self, tokens):
        result = set(
            map(lambda t: t._.stm[0], filter(lambda t: len(t._.stm) > 0, tokens))
        )
        result.discard(None)
        result = list(result)
        result.sort(key=lambda s: s.id)
        return result


class NotNeredException(Exception):
    pass
