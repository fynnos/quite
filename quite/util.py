from spacy.tokens import Span


def span_overlap(a: Span, b: Span):
    a1, a2 = a.start, a.end
    b1, b2 = b.start, b.end
    overlap = max(0, min(a2, b2) - max(a1, b1))
    return overlap


def span_intersection(a: Span, b: Span):
    l, r = max(a.start, b.start), min(a.end, b.end)
    return a.doc[l:r] if r > l else a.doc[0:0]


def span_union(a: Span, b: Span):
    l, r = min(a.start, b.start), max(a.end, b.end)
    return a.doc[l:r]
