#!/usr/bin/env python
from sys import argv
import os

__humans = None
with open(
    os.path.join(os.path.dirname(__file__), "data/mensch-unique.txt"), "r"
) as file:
    __humans = frozenset(file.read().splitlines())


def is_human(word: str):
    return word in __humans


if __name__ == "__main__":
    inputs = argv[1:] if len(argv) > 1 else ["Sprecher"]
    for n in inputs:
        print(n, is_human(n), sep="\t")
