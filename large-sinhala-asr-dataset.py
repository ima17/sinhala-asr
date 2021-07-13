import os
import string
import shutil

import datasets
from datasets.tasks import AutomaticSpeechRecognition


_DATA_CLIPS_URL = "https://www.openslr.org/resources/52/asr_sinhala_{}.zip"

_TRAIN_DATA_URL = "https://raw.githubusercontent.com/keshan/sinhala-asr/main/train.tsv"
_TEST_DATA_URL = "https://raw.githubusercontent.com/keshan/sinhala-asr/main/test.tsv"

_CITATION = """\
 @inproceedings{kjartansson-etal-sltu2018,
    title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese,  Sinhala, Nepali, and Bangladeshi Bengali}},
    author = {Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
    booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
    year  = {2018},
    address = {Gurugram, India},
    month = aug,
    pages = {52--55},
    URL   = {http://dx.doi.org/10.21437/SLTU.2018-11}
  }
"""

_DESCRIPTION = """\
This data set contains ~185K transcribed audio data for Sinhala. The data set consists of wave files, and a TSV file. The file utt_spk_text.tsv contains a FileID, anonymized UserID and the transcription of audio in the file.
The data set has been manually quality checked, but there might still be errors.

See LICENSE.txt file for license information.

Copyright 2016, 2017, 2018 Google, Inc.
"""

_HOMEPAGE = "https://www.openslr.org/52/"

_LICENSE = "https://www.openslr.org/resources/52/LICENSE"

_LANGUAGES = {
    "si": {
        "Language": "Sinhala",
        "Date": "2018",
    },
}


class LargeASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LargeASR."""

    def __init__(self, name, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.language = kwargs.pop("language", None)
        self.date_of_snapshot = kwargs.pop("date", None)
        description = f"Large Sinhala dataset in {self.language} of {self.date_of_snapshot}."
        super(LargeASRConfig, self).__init__(
            name=name, version=datasets.Version("1.0.0", ""), description=description, **kwargs
        )


class LargeASR(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        LargeASRConfig(
            name=lang_id,
            language=_LANGUAGES[lang_id]["Language"],
            date=_LANGUAGES[lang_id]["Date"],
        )
        for lang_id in _LANGUAGES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "x": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "file": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                AutomaticSpeechRecognition(audio_file_path_column="file", transcription_column="sentence")
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_file_urls = [_DATA_CLIPS_URL.format(i) for i in (string.digits + string.ascii_lowercase[:6])]
        dl_paths = dl_manager.download_and_extract(data_file_urls)
        
        # Moving all the downloaded audio clips to one parent folder.
        dirname = os.path.dirname
        for path in dl_paths:
            shutil.copytree(path, dirname(path), dirs_exist_ok=True)
              
        abs_path_to_train_data = dl_manager.download_and_extract(_TRAIN_DATA_URL)
        abs_path_to_test_data = dl_manager.download_and_extract(_TEST_DATA_URL)
        abs_path_to_clips = os.path.dirname(dl_paths[0])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_train_data, "train.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_test_data, "test.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
        ]

    def _generate_examples(self, filepath, path_to_clips):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        path_idx = data_fields.index("file")

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
            headline = lines[0]

            column_names = headline.strip().split("\t")
            assert (
                column_names == data_fields
            ), f"The file should have {data_fields} as column names, but has {column_names}"

            for id_, line in enumerate(lines[1:]):
                field_values = line.strip().split("\t")
                
                # set absolute path for wav audio file
                field_values[path_idx] = os.path.join(path_to_clips, field_values[path_idx])

                # if data is incomplete, fill with empty values
                if len(field_values) < len(data_fields):
                    field_values += (len(data_fields) - len(field_values)) * ["''"]

                yield id_, {key: value for key, value in zip(data_fields, field_values)}