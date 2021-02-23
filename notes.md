# Notes on LDA pipeline development

- I did add ‘’“” as context characters to tokenizer, but that is deprecated because I just replace those before passing data frame to document assembler.

- I did set tokenizer exceptions as in "[a-zA-Z0-9\']\s+sell" to make a unigram of "don't sell", but I handle this differently now using POS tagging. It's a viable option for a simpler and faster pipeline that uses unigrams only.

Also, I initially used a bad regular expression instead of the one below, and I got a very obscure error message while trying to fit the count vectorizer. I later realized I would get the same error message while trying to run .collect on `processed_texts.select(["unigrams"])` (and other data frames), or if I tried to output to Pandas. I could run `processed_texts.show()`, but I couldn't write to JSON. I also did't get the error while using a subsample of the dataset.


- I was using the standard Englished pretrained stopwords cleaner, but it had too large a vocabulary. For instance, it would remove the word "example".


- I decided to scrap the spell checker because the built-in ones don't do a very good job. E.g., Norvig-Sweeting takes "January" to "Manuary", and the Context-Aware DL model takes "Gamestop" to "Jameson". I didn't try very hard to improve on this--maybe I will do later.

