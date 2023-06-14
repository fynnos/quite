import csv
import os
from sys import stdout, stderr, exit, argv, stdin
from typing import List

import spacy
from spacy.tokens import Span
from quite.statement import Statement

language = os.environ.get("LANGUAGE", "de")
if language not in ("de", "en"):
    print("Only 'de' or 'en' are supported languages, not:", language, file=stderr)
    exit(1)

# e.g. 'http://localhost:1080/predictions/tuba10_electra_uncased_512_3' or 'http://localhost:1080/predictions/train_spanbert_large_ml0_d1'
use_coref = bool(os.environ.get("COREF_URL", ""))
# e.g. 'http://localhost:5002/api'
use_external_ner = bool(os.environ.get("NER_URL", ""))
encoding = os.environ.get("ENCODING", "utf-8")
# e.g. en_core_web_md en_core_web_lg en_core_web_sm
model_en = os.environ.get("MODEL_EN", "en_core_web_lg")
# e.g. de_core_news_sm de_core_news_md de_core_news_lg
model_de = os.environ.get("MODEL_DE", "de_core_news_lg")

model = spacy.load(model_de if language == "de" else model_en)

if use_external_ner and language == "de":
    from microner import MicroNer

    micro_ner = MicroNer(model, os.environ.get("NER_URL"))
    model.replace_pipe("ner", micro_ner)
if use_coref:
    from corefserve import CorefServe
    from reducecoref import ReduceCoref

    model.add_pipe("corefserve", config={"url": os.environ.get("COREF_URL")})
    model.add_pipe("reducecoref")

use_iwnlp = language == "de"
if use_iwnlp:
    model.add_pipe(
        "iwnlp",
        config={
            "datapath": os.path.join(
                os.path.dirname(__file__), "data/IWNLP.Lemmatizer_20181001.json"
            )
        },
    )

model.add_pipe(
    "statements",
    config={"language": language, "use_iwnlp": use_iwnlp, "use_morph": True},
)


def formatRef(span: Span):
    return f"{span}({span.start_char}-{span.end_char})"


def writeCsv(statements: List[Statement], name: str, writer):
    for id, subj, verb, text, origin in statements:
        if use_coref and subj is not None and subj._.is_coref:
            refs = (
                "|".join(formatRef(c.main) for c in subj._.corefs)
                if len(subj._.corefs) > 1
                else formatRef(subj._.corefs[0].main)
            )
        else:
            refs = ""
        writer.writerow(
            (
                name,
                subj,
                None if subj is None else subj.start_char,
                None if subj is None else subj.end_char,
                refs,
                verb,
                None if verb is None else verb.doc[verb.i : verb.i + 1].start_char,
                None if verb is None else verb.doc[verb.i : verb.i + 1].end_char,
                text,
                text.start_char,
                text.end_char,
            )
        )


def processFile(filename: str, writer):
    with open(filename, "r", encoding=encoding) as file:
        text = file.read()
    doc = model(text)
    writeCsv(doc._.statements, os.path.basename(filename), writer)


def main():
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print(
            """Usage: quite.py <input files> <output file>

        if the last file exists, it is assumed to be an input file
        if no input files are specified, stdin is used
        if no output file is specified, stdout is used""",
            file=stderr,
        )
        exit(0)
    if os.path.exists(argv[-1]):
        output = stdout
        args = argv[1:]
    else:
        output = open(argv[-1], "w")
        args = argv[1:-1]
    writer = csv.writer(output)
    writer.writerow(
        (
            "filename",
            "subject",
            "subject_start",
            "subject_end",
            "subject references",
            "cue",
            "cue_start",
            "cue_end",
            "quote",
            "quote_start",
            "quote_end",
        )
    )
    if len(args) == 0:
        processFile(stdin.fileno(), writer)
    elif len(args) == 1 and os.path.isdir(args[0]):
        for file in os.listdir(args[0]):
            processFile(os.path.join(args[0], file), writer)
    else:
        for filename in args:
            processFile(filename, writer)
