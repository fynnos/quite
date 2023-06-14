# QUITE - Quotes in Text

Tool to automatically extract (in)direct speech from German and English texts.


## Quick start

`docker run --rm -e LANGUAGE='en' -v /absolute/path/to/input/files/:/inputs fynnos/quite:latest python -m quite /inputs > outputfilename.csv`

Processes all files in the input folder and writes a CSV file as output.
All files must be plain text files (best already cleaned if crawled from the internet) encoded as UTF8.
To select German as language for processing German texts, replace `LANGUAGE='en'` with `LANGUAGE='de'`.


## Output description

The CSV output has the following columns:

`filename,subject,subject_start,subject_end,subject references,cue,cue_start,cue_end,quote,quote_start,quote_end`

* `filename`: name of the input file where the quote was extracted from
* `subject`: speaker of the quote, possibly empty
* `subject_start`: Begin of the subject text span in characters (inclusive)
* `subject_end`: End of the subject text span in characters (exclusive)
* `subject references`: empty unless QUITE is configured to use a compatible co-reference resolution service
* `cue`: cue/trigger/indicator of a speech, usually a verb, possibly empty
* `cue_start`: analogous to `subject_start`
* `cue_end`: analogous to `subject_end`
* `quote`: the extracted quotation, (in)direct speech
* `quote_start`: analogous to `subject_start`
* `quote_end`: analogous to `subject_end`
