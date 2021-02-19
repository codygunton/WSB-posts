from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import (Tokenizer,
                                SentenceDetector,
                                Normalizer,
                                LemmatizerModel,
                                StopWordsCleaner,
                                # NGramGenerator,
                                # PerceptronModel
                                )
from pyspark.ml import Pipeline
import unicodedata

from download_pretrained import PretrainedCacheManager

emoji_ranges = [
        # ranges covering all emojis expressible
        # using only one unicode character.
        '\U0001f300-\U0001f321',
        '\U0001f324-\U0001f393',
        '\U0001f39e-\U0001f3f0',
        '\U0001f400-\U0001f4fd',
        '\U0001f4ff-\U0001f53d',
        '\U0001f550-\U0001f567',
        '\U0001f5fa-\U0001f64f',
        '\U0001f680-\U0001f6c5',
        '\U0001f90c-\U0001f93a',
        '\U0001f947-\U0001f978',
        '\U0001f97a-\U0001f9cb',
        '\U0001f9cd-\U0001f9ff',
        '\U0001fa90-\U0001faa8'
]


def get_unigram_pipeline_components():
    # get acche manager to avoid repeated downloads
    cache_manager = PretrainedCacheManager()
    cache_manager.get_pretrained_components()
    # this is a dict with entries as in
    # ('lemmatizer', path-to-downloaded-unzipped-lemmatizer)
    pretrained_components = cache_manager.pretrained_components

    # get document assembler
    assembler = DocumentAssembler()

    # get sentence detector
    sentence_detector = SentenceDetector()

    # build tokenizer
    tokenizer = Tokenizer()
    # add ['‘', '’', '“', '”'] as context characters
    # doing this in a verbose way because it may be clarifying.
    char_names = ['LEFT SINGLE QUOTATION MARK',
                  'RIGHT SINGLE QUOTATION MARK',
                  'LEFT DOUBLE QUOTATION MARK',
                  'RIGHT DOUBLE QUOTATION MARK']
    for name in char_names:
        tokenizer.addContextChars(unicodedata.lookup(name))
    # now set exceptions.
    # 1) to preserve word preceding "sell" and "hold",
    #    so that, e.g., "don't sell" is not normalized to "sell".
    # 2) don't split "game stop" (...or "game stonk")
    # 3) if an emoji is repeated with spaces, don't split
    #    so we can normalize later
    tokenizer.setExceptions(["\S+ sell",
                             "\S+ hold",
                             "game [stop|stonk]",
                             "\U0001f680\s+", # rocket ship
                             "\U0001f48e\U0001f64c\s+"
                            ])
    tokenizer.setCaseSensitiveExceptions(True)

    # built stopwords cleaner
    stopwords_cleaner = (
        StopWordsCleaner()
        # on second thought, maybe the larger list of stopwords that I
        # was downloading was too expansive. e.g., it has the word "example"
        # .load(pretrained_components["stopwords"])
        .setCaseSensitive(False)
    )
    # for each stopword involving an apostrophe ('), append
    # a version of the stopword using the character (’) instead,
    # and a version with the apostrophe missing
    char = unicodedata.lookup('APOSTROPHE')
    replacement = unicodedata.lookup('RIGHT SINGLE QUOTATION MARK')
    stopwords = stopwords_cleaner.getStopWords()
    stopwords += ["y'all", "yall"]
    for s in stopwords_cleaner.getStopWords():
        if char in s:
            stopwords.append(s.replace(char, replacement))
            stopwords.append(s.replace(char, ""))
    stopwords.sort()
    stopwords_cleaner.setStopWords(stopwords)

    # build lemmatizer
    lemmatizer = (
        LemmatizerModel().load(pretrained_components["lemmatizer"])
    )

    # build normalizer
    normalizer = (
        Normalizer()
        .setLowercase(True)
    )
    # this does not keep all emojis, but it keeps a lot of them.
    # for instance, it does not distinguish skin color, but it has
    # enough characters to express the Becky emoji.
    keeper_regex = ''.join([
        '[^0-9A-Za-z$&%=',
        # special characters for Becky
        '\u200d\u2640\u2641\u26A5\ufe0f',
        ''.join(emoji_ranges),
        ']'
    ])
    normalizer.setCleanupPatterns([keeper_regex,
                                   'http.*'])

    # build finisher
    finisher = Finisher()

    return (assembler, sentence_detector, tokenizer,
            stopwords_cleaner, lemmatizer, normalizer, finisher)


def get_Ngram_pipeline_components(N=2):
    return


def build_unigram_pipeline(pipeline_components=None):
    # get_pipeline_components
    if not pipeline_components:
        _ = get_unigram_pipeline_components()
    else:
        _ = pipeline_components
    (assembler,
     sentence_detector,
     tokenizer,
     stopwords_cleaner,
     lemmatizer,
     normalizer,
     finisher) = _

    # assemble the pipeline
    (assembler
     .setInputCol('text')
     .setOutputCol('document'))

    (sentence_detector
     .setInputCols(['document'])
     .setOutputCol('sentences'))

    (tokenizer
     .setInputCols(['sentences'])
     .setOutputCol('tokenized'))

    (stopwords_cleaner
     .setInputCols(['tokenized'])
     .setOutputCol('cleaned'))

    (lemmatizer
     .setInputCols(['cleaned'])
     .setOutputCol('lemmatized'))

    (normalizer
     .setInputCols(['lemmatized'])
     .setOutputCol('unigrams'))

    (finisher
     .setInputCols(['unigrams']))

    pipeline = (Pipeline()
                .setStages([assembler,
                            sentence_detector,
                            tokenizer,
                            stopwords_cleaner,
                            lemmatizer,
                            normalizer,
                            finisher]))

    # to do: try LightPipeline as in
    # https://nlp.johnsnowlabs.com/docs/en/concepts#lightpipeline

    return pipeline
