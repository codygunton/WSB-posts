from sparknlp.base import (DocumentAssembler,
                           Finisher)

from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import types as T
import pyspark.sql.functions as F

from download_pretrained import PretrainedCacheManager      
        

def preprocess_texts(df):
    texts = (
        # to do: make its own module extending spark Transformer
        # to do: learn to efficiently fuse these in Spark SQL.
        #
        # I would like to do mild emoji normalization followed by
        # fine tuning a language model to learn the meaning of a few
        # of the major emojis, but for now I just drop them and use
        # a sentence embedding that probably doesn't benefit from 
        # the presence of emojis here.
        df
        .withColumn("text",
                      F.regexp_replace("text",
                                       "http\S+|\&*\#*x200B\S*|\\*",
                                       " " # can't replace with ""
                                       ))
        # this replacement leads, eventually, to the chosen expressions
        # being deleted later on by other components' handling of white
        # space and periods.
        .withColumn("text",
                      F.regexp_replace("text",
                                       "[\\n\\r]|\\- |[\.]{2,}",
                                       ". "))
        # could probably combine the next two operations
        .withColumn("text", 
                      F.regexp_replace("text", "[“”]", "\""))
        .withColumn("text", 
                    F.regexp_replace("text", "[‘’]", "\'"))
        
        .select(["id", 
                 "text"])
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
    # note: shrink cleanup mode seems to lose some sentences
    # anyway, sentencer seems to make document cleanup redundant

    # get sentence detector
    # this currently turns an ellipis into a period.
    sentence_detector = (
        SentenceDetector()
        .setMinLength(3)
        .setExplodeSentences(True)
    )

    # build normalizer
    document_normalizer = DocumentNormalizer()
    # this is inefficient now--could go in the earlier preprocessing.
    # will leave it for now to avoid accidentally breaking anything.
    keeper_regex = ''.join(['[^0-9A-Za-z\$\&%\.,\?!\'‘’\"“”]'])
    document_normalizer.setPatterns([keeper_regex])

    
    # build finisher
    finisher = Finisher().setIncludeMetadata(True)

    return (assembler,
            sentence_detector,
            document_normalizer,
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
                            finisher
                            ]))

    return pipeline