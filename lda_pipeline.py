from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import (Tokenizer,
                                Normalizer,
                                LemmatizerModel,
                                StopWordsCleaner,
                                NGramGenerator,
                                PerceptronModel)
from pyspark.ml import Pipeline

# from spacy.lang.en.stop_words import STOP_WORDS
# STOP_WORDS = list(STOP_WORDS)
# print(STOP_WORDS)


def build_pipeline():
    assembler = (
        DocumentAssembler()
        .setInputCol('text')
        .setOutputCol('document')
    )

    tokenizer = (
        Tokenizer()
        .setInputCols(['document'])
        .setOutputCol('tokenized')
    )

    stopwords_cleaner = (
        StopWordsCleaner.pretrained("stopwords_en", "en")
        .setInputCols(['tokenized'])
        .setOutputCol('cleaned')
        .setCaseSensitive(False)
    )

    # stopwords_cleaner = (
    #     StopWordsCleaner()
    #     .setInputCols(['tokenized'])
    #     .setOutputCol('cleaned')
    #     .setStopWords(STOP_WORDS)
    #     .setCaseSensitive(False)
    # )

    lemmatizer = (
        LemmatizerModel.pretrained()
        .setInputCols(['cleaned'])
        .setOutputCol('lemmatized')
    )

    normalizer = (
        Normalizer()
        .setInputCols(['lemmatized'])
        .setOutputCol('normalized')
        .setLowercase(True)
        .setCleanupPatterns(['[^A-Za-z]',
                             'http.*'])
    )

    # ngrammer = (
    #     NGramGenerator()
    #     .setInputCols([''])
    #     .setOutputCol('ngrams')
    #     .setN(3)
    #     .setEnableCumulative(True)
    #     .setDelimiter('_')
    # )

    # pos_tagger = (
    #     PerceptronModel.pretrained('pos_anc')
    #     .setInputCols(['document', 'lemmatized'])
    #     .setOutputCol('pos')
    # )

    finisher = (
        Finisher()
        .setInputCols(['normalized',
                       # 'pos'
                       ])
    )

    pipeline = (Pipeline()
                .setStages([assembler,
                            tokenizer,
                            stopwords_cleaner,
                            lemmatizer,
                            normalizer,
                            # pos_tagger,
                            # ngrammer,
                            finisher]))

    # to do: try LightPipeline as in
    # https://nlp.johnsnowlabs.com/docs/en/concepts#lightpipeline

    return pipeline
