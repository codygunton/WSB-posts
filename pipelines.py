from sparknlp.base import (DocumentAssembler,
                           Finisher)
# from sparknlp.annotator import (RegexMatcher,
#                                 Tokenizer,
#                                 SentenceDetector,
#                                 SentenceDetectorDLApproach,
#                                 Normalizer,
#                                 DocumentNormalizer,
#                                 LemmatizerModel,
#                                 StopWordsCleaner,
#                                 NorvigSweetingModel,
#                                 UniversalSentenceEncoder,
#                                 NGramGenerator,
#                                 # PerceptronModel
#                                 )
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import unicodedata

from download_pretrained import PretrainedCacheManager

emoji_ranges = [
        # special characters for Becky
        '\u200d\u2640\u2641\u26A5\ufe0f',
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


def write_emoji_matcher_rules():
    # generate matcher rules
    emojis = ''.join(emoji_ranges)
    # regex = f"[{emojis}|{emojis}\s+]+"
    regex = f"[{emojis}]"
    matcher_rule = regex+"~emojis"
    with open("emoji_matcher_rules.csv", "w") as fobj:
        fobj.writelines([matcher_rule])


def get_bowbae_components():
    # get cache mananager to avoid repeated downloads
    cache_manager = PretrainedCacheManager()
    cache_manager.get_pretrained_components()
    # this is a dict with entries as in
    # ('lemmatizer', path-to-downloaded-unzipped-lemmatizer)
    pretrained_components = cache_manager.pretrained_components

    # get document assemblers
    assembler = DocumentAssembler()
    assembler_no_emojis = DocumentAssembler()

    # get matcher
    write_emoji_matcher_rules()
    emoji_matcher = RegexMatcher()
    emoji_matcher.setExternalRules("emoji_matcher_rules.csv",
                                   delimiter="~")


      # build normalizer
    document_normalizer = (
        DocumentNormalizer()
        .setLowercase(True)
    )
    # this does not keep all emojis, but it keeps a lot of them.
    # for instance, it does not distinguish skin color, but it has
    # enough characters to express the Becky emoji.
    keeper_regex = ''.join([
        '[^0-9A-Za-z$&%\.,?!\'‘’\"“”',
#         ''.join(emoji_ranges),
        ']' 
    ])
    document_normalizer.setPatterns([keeper_regex,
                                     'http.*',])    

    
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
                             "\S+ buy",
                             "game [stop|stonk]",
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
        ']'
    ])
    normalizer.setCleanupPatterns([keeper_regex, "\'\"",
                                   'http.*'])
#     # decided to scrap:
#     # Norvig-Sweeting takes "January" to "manuary"
#     # Context-Aware DL model takes "Gamestop" to "Jameson".
#     spell_checker = (
#         NorvigSweetingModel()
# #         .pretrained()
#         .load(pretrained_components["spell_checker"])
#     )

    ngrammer = (
        NGramGenerator()
        .setN(2)
        .setDelimiter("_")
    )
    
    pos_tagger = (
        PerceptronModel()
        .load(pretrained_components["pos_tagger"])
    )
    
    chunker = (
        Chunker()
        .setRegexParsers(['<JJ>+<NN>', '<NN>+<NN>'])
    )
    
    # build finisher
    finisher = Finisher()

    return (assembler,
            assembler_no_emojis,
            emoji_matcher,
            document_normalizer,
            sentence_detector,
            tokenizer,
            stopwords_cleaner,
            lemmatizer, 
            normalizer,
#             spell_checker,
            ngrammer,
            pos_tagger,
            chunker,
            finisher)


# bag of words bag of emojis pipeline
def build_bowbae_pipeline(pipeline_components=None):
    # get_pipeline_components
    if not pipeline_components:
        _ = get_bowbae_components()
    else:
        _ = pipeline_components
    (assembler,
     assembler_no_emojis,
     emoji_matcher,
     document_normalizer,
     sentence_detector,
     tokenizer,
     stopwords_cleaner,
     lemmatizer,
     normalizer,
#      spell_checker,
     ngrammer,
     pos_tagger,
     chunker,
     finisher) = _

    # assemble the pipeline
    (assembler
     .setInputCol('text')
     .setOutputCol('document_emojis'))

    (emoji_matcher
     .setInputCols(["document_emojis"])
     .setOutputCol("emojis"))
    
    (assembler_no_emojis
     .setInputCol('text_no_emojis')
     .setOutputCol('document'))
    
    (document_normalizer
     .setInputCols(['document'])
     .setOutputCol('normalized_document'))
    

    (sentence_detector
     .setInputCols(['normalized_document'])
     .setOutputCol('sentences'))

    (tokenizer
     .setInputCols(['sentences'])
     .setOutputCol('tokenized'))

    (stopwords_cleaner
     .setInputCols(['tokenized'])
     .setOutputCol('cleaned'))

    (lemmatizer
     .setInputCols(['cleaned'])
     .setOutputCol('unigrams'))

    (normalizer
     .setInputCols(['cleaned'])
     .setOutputCol('unigrams'))
    
#     (spell_checker
#      .setInputCols(['normalized'])
#      .setOutputCol('unigrams'))
    
    (ngrammer
     .setInputCols(['unigrams'])
     .setOutputCol('naive_ngrams'))
    
    (pos_tagger
     .setInputCols(['unigrams', 'sentences'])
     .setOutputCol('pos_tags'))
    
    (chunker
     .setInputCols(['normalized_document', 'pos_tags'])
     .setOutputCol('ngrams')
    )
    
    (finisher
     .setInputCols(['tokenized',
                    'emojis', 
                    'unigrams', 
                    'naive_ngrams',
                    'pos_tags',
                    'ngrams'
                   ]))

    pipeline = (Pipeline()
                .setStages([assembler,
                            emoji_matcher,
                            assembler_no_emojis,
                            document_normalizer,
                            sentence_detector,
                            tokenizer,
                            stopwords_cleaner,
                            lemmatizer,
                             normalizer,
#                             spell_checker,
                            ngrammer,
                            pos_tagger,
                            chunker,
                            finisher]))

    # to do: try LightPipeline as in
    # https://nlp.johnsnowlabs.com/docs/en/concepts#lightpipeline

    return pipeline

def get_new_components():
    # get cache mananager to avoid repeated downloads
    cache_manager = PretrainedCacheManager()
    cache_manager.get_pretrained_components()
    # this is a dict with entries as in
    # ('lemmatizer', path-to-downloaded-unzipped-lemmatizer)
    pretrained_components = cache_manager.pretrained_components

    # get document assemblers
    assembler = DocumentAssembler()
    # note: shrink cleanup mode seems to lose some sentnces
    # anyway, sentencer seems to make document cleanup redundant

    # get sentence detector
    # this currently turns an ellipis into a period.
    sentence_detector = (
        SentenceDetector()
        .setMinLength(3)
        .setExplodeSentences(True)
    )
    #.setExplodeSentences(True)

    # build normalizer
    document_normalizer = (
        DocumentNormalizer()
        # .setLowercase(True)
    )
    # this does not keep all emojis, but it keeps a lot of them.
    # for instance, it does not distinguish skin color, but it has
    # enough characters to express the Becky emoji.
    keeper_regex = ''.join([
        '[^0-9A-Za-z$&%\.,?!\'‘’\"“”',
        ''.join(emoji_ranges),
        ']' 
    ])
    document_normalizer.setPatterns([keeper_regex,
                                     'http.*',])

#     USE = (
#         UniversalSentenceEncoder()
#         .load(pretrained_components["USE"])
#     )
    
    # build finisher
    finisher = Finisher()

    return (assembler,
            sentence_detector,
            document_normalizer,
#             USE,
            finisher)


# bag of words bag of emojis pipeline
def build_new_pipeline(pipeline_components=None):
    # get_pipeline_components
    if not pipeline_components:
        _ = get_new_components()
    else:
        _ = pipeline_components
    (assembler,
     sentence_detector,
     document_normalizer,
     # USE,
     finisher
    ) = _

    # assemble the pipeline
    (assembler
     .setInputCol('text')
     .setOutputCol('document'))

    (document_normalizer
     .setInputCols(['document'])
     .setOutputCol('normalized_document'))

    (sentence_detector
     .setInputCols(['normalized_document'])
     .setOutputCol('sentences'))

#     (USE
#      .setInputCols(['sentences'])
#      .setOutputCol('use_embedding')

    (finisher
     .setInputCols(['sentences']))

    pipeline = (Pipeline()
                .setStages([assembler,
                            document_normalizer,
                            sentence_detector,
#                             USE, 
#                             finisher
                            ]))

    # to do: try LightPipeline as in
    # https://nlp.johnsnowlabs.com/docs/en/concepts#lightpipeline

    return pipeline



