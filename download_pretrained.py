# I think if I rewrote the pipeline in Scala I could use the
# ResourceDownloader class to manage cashed pretrained, but
# the Python interface seemes to be really limited at the moment.
# This file gives a simple way of managinging the model cache
# to avoid downloading the files every time we instantiate them
# pipeline components.

# The links to the resources are found manually using the models db
# maintained by John Snow Labs. Keeping the defauled chache dir,
# unless the aws address changes, it should also work to download
# resources once using the .pretrained methods LemmatizerModel, etc.

import os
import requests
import zipfile


class PretrainedCacheManager:
    def __init__(self, cache_dir=None):
        if not cache_dir:
            cache_dir = os.environ["HOME"] + "/cache_pretrained"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.urls = {}
        base_url = ("https://s3.amazonaws.com/" +
                    "auxdata.johnsnowlabs.com/public/models")
        self.urls["lemmatizer"] = (base_url +
                                   "/lemma_antbnc_en_2.0.2_2.4_" +
                                   "1556480454569.zip")
        self.urls["stopwords"] = (base_url +
                                  "/stopwords_en_en_2.5.4_2.4_" +
                                  "1594742439135.zip")
#         self.urls["spell_checker"] = (base_url +
#                                   "/spellcheck_norvig_en_2.1.0_2.4_" +
#                                   "1563017660080.zip")
#         self.urls["USE"] = (base_url + 
#                             "/tfhub_use_en_2.4.0_2.4_" +
#                                   "1587136330099.zip")
        self.urls["pos_tagger"] = (base_url +
                                  "/pos_anc_en_2.0.2_2.4_" +
                                  "1556659930154.zip")

        # this will be a dict with entries as in
        # ('lemmatizer', path-to-downloaded-unzipped-lemmatizer)
        self.pretrained_components = {}

    def get_pretrained_components(self):
        for key in self.urls:
            component_name = self.urls[key].split("/")[-1]
            zip_path = f"{self.cache_dir}/{component_name}"
            target_path = zip_path.replace(".zip", "/")

            self.pretrained_components[key] = target_path
            if not os.path.exists(target_path):
                print(f"Downloading {key}.")
                url = self.urls[key]
                r = requests.get(url)
                with open(zip_path, "wb") as fobj:
                    for chunk in r.iter_content(chunk_size=128):
                        fobj.write(chunk)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(target_path.replace(".zip", "/"))
                os.remove(zip_path)
            # else:
            #     print(f"{key.title()} resource is already downloaded.")
