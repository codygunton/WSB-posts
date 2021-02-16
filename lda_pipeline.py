from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import (Tokenizer,
                                Normalizer,
                                LemmatizerModel,
                                StopWordsCleaner,
                                # NGramGenerator,
                                # PerceptronModel
                                )
from pyspark.ml import Pipeline
import unicodedata


# define component pieces here to avoid re-downloading
# the pretrained components every time
# (TODO: figure how to cache downloaded models (...with Python?))

assembler = DocumentAssembler()

tokenizer = Tokenizer()

char_names = ['LEFT SINGLE QUOTATION MARK',
              'RIGHT SINGLE QUOTATION MARK',
              'LEFT DOUBLE QUOTATION MARK',
              'RIGHT DOUBLE QUOTATION MARK']
for name in char_names:
    tokenizer.addContextChars(unicodedata.lookup(name))


stopwords_cleaner = (
    StopWordsCleaner.pretrained("stopwords_en", "en")
    .setCaseSensitive(False)
)

char = unicodedata.lookup('APOSTROPHE')
replacement = unicodedata.lookup('RIGHT SINGLE QUOTATION MARK')
stopwords = stopwords_cleaner.getStopWords()
for s in stopwords_cleaner.getStopWords():
    if char in s:
        stopwords.append(s.replace(char, replacement))
stopwords.sort()
stopwords_cleaner.setStopWords(stopwords)


lemmatizer = LemmatizerModel.pretrained()

normalizer = (
    Normalizer()
    .setLowercase(True)
    .setCleanupPatterns(['[^0-9A-Za-z$&%]',
                         'http.*'])
)

finisher = Finisher()


def build_pipeline():
    (assembler
     .setInputCol('text')
     .setOutputCol('document'))

    (tokenizer
     .setInputCols(['document'])
     .setOutputCol('tokenized'))

    (stopwords_cleaner
     .setInputCols(['tokenized'])
     .setOutputCol('cleaned')
     .setCaseSensitive(False))

    (lemmatizer
     .setInputCols(['cleaned'])
     .setOutputCol('lemmatized'))

    (normalizer
     .setInputCols(['lemmatized'])
     .setOutputCol('normalized'))

    (finisher
     .setInputCols(['normalized']))

    pipeline = (Pipeline()
                .setStages([assembler,
                            tokenizer,
                            stopwords_cleaner,
                            lemmatizer,
                            normalizer,
                            finisher]))

    # to do: try LightPipeline as in
    # https://nlp.johnsnowlabs.com/docs/en/concepts#lightpipeline

    return pipeline
