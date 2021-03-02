from sparknlp.base import (DocumentAssembler,
                           Finisher,
                           Chunk2Doc)

from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import types as T
import pyspark.sql.functions as F

from download_pretrained import PretrainedCacheManager

emojis_regex = ( 
    # first the Becky emoji
    '\U0001f471\u200d\u2640\ufe0f|' +
    '[' +
    "".join([
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
    ])
    + 
    ']'
)


def write_emoji_matcher_rules():
    # generate matcher rules
    matcher_rule = emojis_regex+"~emojis"
    with open("emoji_matcher_rules.csv", "w") as fobj:
        fobj.writelines([matcher_rule])
        
        

def preprocess_texts(df):
    texts = (
        # to do: make its own module extending spark Transformer
        df.withColumn("text_no_emojis",
                      F.regexp_replace("text",
                                       # replacing with "" is bad
                                       emojis_regex, " "))
        .withColumn("text_no_emojis", 
                      F.regexp_replace("text_no_emojis", "[“”]", "\""))
        .withColumn("text_no_emojis", 
                    F.regexp_replace("text_no_emojis", "[‘’]", "\'"))
        # to keep positions of emojis (not necessary, currently)
        .select(["id", 
                 "timestamp", 
                 "text", 
                 "text_no_emojis"])
    )    
    return texts


# Pipeline for cleaning text before processing with pretrained embeddings.
# This is under construction.
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
    keeper_regex = ''.join(['[^0-9A-Za-z\$\&%\.,\?!\'‘’\"“”]'])
    document_normalizer.setPatterns([keeper_regex,
                                     'http\S+', 
                                     '\&*\#*x200B\S*'])
    
#     USE = (
#         UniversalSentenceEncoder()
#         .load(pretrained_components["USE"])
#     )
    
    # build finisher
    finisher = Finisher().setIncludeMetadata(True)

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

    (finisher
     .setInputCols(['sentences']))

    pipeline = (Pipeline()
                .setStages([assembler,
                            document_normalizer,
                            sentence_detector,
#                             USE, 
                            finisher
                            ]))

    return pipeline


def embedding_preproc_finisher(df):    
    output = (df
              .select("text", "unigrams")
              # explode unigrams columsn into rows
              .withColumn("exploded", 
                          F.explode(df['unigrams']))
              # select only columns of intereest
              .select("text", "exploded.result", "exploded.metadata"))

    output = (output
              .select("text",
                      output["result"].alias("unigram"),
                      # metadata is a map as in sentence -> 0 (or 1, 2,...)
                      # just keep the value
                      F.element_at(output.metadata, 'sentence')
                      .alias("sentence"))
              # concatenate unigrams to ngrams
              .groupby('sentence', 'text')
              .agg(F.concat_ws("_", F.collect_list('unigram'))
                    .alias("ngram"))
              # collect ngrams into arrays, one for each text
              .groupby('text')
              .agg(F.collect_list('ngram')
                    .alias("finished_ngrams")))
    
    # adjoin emojis
    emoji_finisher = Finisher().setInputCols("emojis")
    emoji_df = emoji_finisher.transform(df)
    output = (output
              .join(emoji_df.select("text", "finished_emojis"), 
                    "text"))
    
    return output
