from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import (Tokenizer,
                                Normalizer,
                                LemmatizerModel,
                                StopWordsCleaner,
                                NGramGenerator,
                                PerceptronModel)
from pyspark.ml import Pipeline

from spacy.lang.en.stop_words import STOP_WORDS

STOP_WORDS = list(STOP_WORDS)


def build_pipeline():
    assembler = (
        DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
    )

    tokenizer = (
        Tokenizer()
        .setInputCols(['document'])
        .setOutputCol('tokenized')
    )

    normalizer = (
        Normalizer()
        .setInputCols(['tokenized'])
        .setOutputCol('normalized')
        .setLowercase(True)
        .setCleanupPatterns(['[^A-Za-z]',
                             'http.*'])
    )

    lemmatizer = (
        LemmatizerModel.pretrained()
        .setInputCols(['normalized'])
        .setOutputCol('lemmatized')
    )

    stopwords_cleaner = (
        StopWordsCleaner()
        .setInputCols(['lemmatized'])
        .setOutputCol('unigrams')
        .setStopWords(STOP_WORDS)
    )

    # ngrammer = (
    #     NGramGenerator()
    #     .setInputCols(['lemmatized'])
    #     .setOutputCol('ngrams')
    #     .setN(3)
    #     .setEnableCumulative(True)
    #     .setDelimiter('_')
    # )

    pos_tagger = (
        PerceptronModel.pretrained('pos_anc')
        .setInputCols(['document', 'lemmatized'])
        .setOutputCol('pos')
    )

    finisher = (
        Finisher()
        .setInputCols(['unigrams',
                       'pos'])
    )

    pipeline = (Pipeline()
                .setStages([assembler,
                            tokenizer,
                            normalizer,
                            lemmatizer,
                            stopwords_cleaner,
                            pos_tagger,
                            # ngrammer,
                            finisher]))

    return pipeline
