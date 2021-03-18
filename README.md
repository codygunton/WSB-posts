As part of [a project to digest an analyze posts on the Subreddit r/WallStreetBets](https://github.com/codygunton/WSB-posts), I have created plots like the following one. The posts were cleaned to resemble more standard English using [Spark NLP](https://nlp.johnsnowlabs.com/). They were then embedded in 768 embeddings using the pretrained Tensorflow model [Universal Sentence Encoder CMLM](https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/). From there, the dimension was reduced (in this plot, to five) using [UMAP](https://umap-learn.readthedocs.io/en/latest/). With some hyperparameter tuning, the results were clustered using [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html). The resuts were then projeted into two dimensions using UMAP and plotted in Bokeh. This work, along with much more, is available at the public link above.


<iframe src="assets/wsb_emb.html" 
        sandbox="allow-same-origin allow-scripts" 
        width="100%" 
        height="660" 
        scrolling="no" 
        seamless="seamless" 
        frameborder="0"> </iframe> 
