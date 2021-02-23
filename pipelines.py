from sparknlp.base import (DocumentAssembler,
                           Finisher)

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


def get_lda_components():
    # get cache mananager to avoid repeated downloads
    cache_manager = PretrainedCacheManager()
    cache_manager.get_pretrained_components()
    # this is a dict with entries as in
    # ('lemmatizer', path-to-downloaded-unzipped-lemmatizer)
    pretrained_components = cache_manager.pretrained_components

    # document assemblers
    assembler = DocumentAssembler()
    assembler_no_emojis = DocumentAssembler()

    # get regex matcher for emojis
    write_emoji_matcher_rules()
    emoji_matcher = RegexMatcher()
    emoji_matcher.setExternalRules("emoji_matcher_rules.csv",
                                   delimiter="~")

    # build document normalizer. Will normalize tokens later.
    document_normalizer = (
        DocumentNormalizer()
        .setLowercase(True)
    )
    # this does not keep all emojis, but it keeps a lot of them.
    # for instance, it does not distinguish skin color, but it has
    # enough characters to express the Becky emoji.
    keeper_regex = ''.join(['[^0-9A-Za-z\$\&%\.,\?!\'‘’\"“”]'])
    document_normalizer.setPatterns([keeper_regex,
                                     'http.*'])    
    
    # get sentence detector. This is  used for POS tagging.
    sentence_detector = SentenceDetector()

    # build tokenier
    tokenizer = (
        Tokenizer()
        .setInfixPatterns(  # didn't try to be exhaustive here.
        ['(\w+)(n\'t)',     # split wasn't |-> was n't
         '(\w+)(\'m)',      #         I'm  |-> I 'm      
         '(\w+)(\'s)'       #       she's  |-> she 's
         ])
    )
    
    # build lemmatizer. dict from:
    #  https://raw.githubusercontent.com/mahavivo/\
    #          vocabulary/master/lemmas/AntBNC_lemmas_ver_001.txt
    lemmatizer = (
        Lemmatizer()
        .setDictionary("./AntBNC_lemmas_ver_001.txt",
                       value_delimiter ="\t",
                       key_delimiter = "->")
        # .load(pretrained_components["lemmatizer"])
    )

    # build token normalizer
    normalizer = Normalizer()
    keeper_regex = ''.join(['[^0-9A-Za-z\$&%=]'])    
    normalizer.setCleanupPatterns([keeper_regex])

    # build stopwords cleaner
    stopwords_cleaner = (
        StopWordsCleaner()
        .setCaseSensitive(False)
    )    
    

    # build n-gram generator.
    ngrammer = (
        NGramGenerator()
        .setN(2)
        .setDelimiter("_")
    )

    # build POS tagger. I wish there were one trained on
    # social media posts, but this doesn't seem to do so badly.
    pos_tagger = (
        PerceptronModel()
        .load(pretrained_components["pos_tagger"])
    )


    # choose which sequences of POS tags can be used to generate n-grams.
    # I choose a large list for purposes of illustraiton, but I would advocate
    # to be more focused in practice, depending on particular objectives.
    chunker = (
        Chunker()
        .setRegexParsers(['<JJ>+<NN>',            # adjective-noun
                          '<NN>+<NN>',            # noun-noun
                          '<MD>+<RB>*<VB>',       # [should, not, sell]
                          '<VBP>*<RB>*<VB>+<NN>'  # [do*, not*, sell, gme]
                          ])
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
            ngrammer,
            pos_tagger,
            chunker,
            finisher)


# bag of words bag of emojis pipeline
def build_lda_pipeline(pipeline_components=None):
    # get_pipeline_components
    if not pipeline_components:
        _ = get_lda_components()
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
     .setInputCols(['normalized_document'])
     .setOutputCol('tokenized'))

    (lemmatizer
     .setInputCols(['tokenized'])
     .setOutputCol('lemmatized'))

    (normalizer
     .setInputCols(['lemmatized'])
     .setOutputCol('unigrams'))

    (stopwords_cleaner
     .setInputCols(['unigrams'])
     .setOutputCol('cleaned_unigrams'))
    
    
    (ngrammer
     .setInputCols(['cleaned_unigrams'])
     .setOutputCol('naive_ngrams'))
    
    (pos_tagger
     .setInputCols(['sentences', 'unigrams'])
     .setOutputCol('pos_tags'))
    
    (chunker
     .setInputCols(['sentences', 'pos_tags'])
     .setOutputCol('ngrams')
    )

    (finisher
     .setInputCols(['tokenized',
                    'cleaned_unigrams',
                    'lemmatized',
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
                            lemmatizer,
                            normalizer,
                            stopwords_cleaner,
                            ngrammer,
                            pos_tagger,
                            chunker,
                            finisher]))

    return pipeline


# Pipeline for cleaning text before processing with pretrained embeddings.

def get_embedding_preproc_components():
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


def build_embedding_preproc_pipeline(pipeline_components=None):
    # get_pipeline_components
    if not pipeline_components:
        _ = get_embedding_preproc_components()
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



