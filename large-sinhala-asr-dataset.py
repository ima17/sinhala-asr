import os

import datasets
from datasets.tasks import AutomaticSpeechRecognition


_DATA_URL = ".tar.gz"

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
This data set contains transcribed audio data for Sinhala. The data set consists of wave files, and a TSV file. The file utt_spk_text.tsv contains a FileID, anonymized UserID and the transcription of audio in the file.
The data set has been manually quality checked, but there might still be errors.

See LICENSE.txt file for license information.

Copyright 2016, 2017, 2018 Google, Inc.
"""

_HOMEPAGE = "https://www.openslr.org/52/"

_LICENSE = "https://www.openslr.org/resources/52/LICENSE"

_LANGUAGES = {
    "si": {
        "Language": "Sinhala",
        "Date": "2020-12-11",
        "Size": "39 MB",
        "Version": "si_1h_2020-12-11",
        "Validated_Hr_Total": 0.05,
        "Overall_Hr_Total": 1,
        "Number_Of_Voice": 14,
    },
}


class LargeASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LargeASR."""

    def __init__(self, name, sub_version, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.sub_version = sub_version
        self.language = kwargs.pop("language", None)
        self.date_of_snapshot = kwargs.pop("date", None)
        self.size = kwargs.pop("size", None)
        self.validated_hr_total = kwargs.pop("val_hrs", None)
        self.total_hr_total = kwargs.pop("total_hrs", None)
        self.num_of_voice = kwargs.pop("num_of_voice", None)
        description = f"Large Sinhala dataset in {self.language} version {self.sub_version} of {self.date_of_snapshot}. The dataset comprises {self.validated_hr_total} of validated transcribed speech data from {self.num_of_voice} speakers. The dataset has a size of {self.size}"
        super(LargeASRConfig, self).__init__(
            name=name, version=datasets.Version("1.0.0", ""), description=description, **kwargs
        )


class LargeASR(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        LargeASRConfig(
            name=lang_id,
            language=_LANGUAGES[lang_id]["Language"],
            sub_version=_LANGUAGES[lang_id]["Version"],
            date=_LANGUAGES[lang_id]["Date"],
            size=_LANGUAGES[lang_id]["Size"],
            val_hrs=_LANGUAGES[lang_id]["Validated_Hr_Total"],
            total_hrs=_LANGUAGES[lang_id]["Overall_Hr_Total"],
            num_of_voice=_LANGUAGES[lang_id]["Number_Of_Voice"],
        )
        for lang_id in _LANGUAGES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "x": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "full": datasets.Value("string"),
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
        # dl_path = dl_manager.download_and_extract(_DATA_URL)
        # abs_path_to_data = os.path.join(dl_path, "cv-corpus-6.1-2020-12-11", self.config.name)
        # abs_path_to_clips = os.path.join(abs_path_to_data, "clips")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "train.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "test.tsv"),
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