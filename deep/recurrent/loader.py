import functools
import os
import os.path
import re
from typing import Dict, Literal, Set, Tuple

import librosa
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset


class SpeechCommandsDataset(Dataset):
    CLASSES = ("one", "two", "three")
    FOLDER = "data_speech_commands_v0.02/"

    FILEPATH_RE = re.compile(r"^(?P<class>one|two|three)\/(?P<file>\w+\.\w+)$")
    SplitFiles = Dict[str, Set[str]]

    TEST_LIST = "testing_list.txt"
    VAL_LIST = "validation_list.txt"

    def __init__(
        self, path: str, split: Literal["train", "val", "test"]
    ) -> None:
        self.path = path

        self.files = []
        self.labels = []

        files_by_class = self.files_by_class(split)

        for _class, files in files_by_class.items():
            label = self.CLASSES.index(_class)

            self.files.extend([self.__path(_class, f) for f in files])
            self.labels.extend([label] * len(files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        wav_sample = wavfile.read(self.files[idx])[1]

        sample = np.zeros((16000,), dtype=np.float32)
        sample[: wav_sample.size] = wav_sample
        sample = sample.reshape((16000,))

        sample = (
            librosa.feature.mfcc(
                y=sample, sr=16000, hop_length=512, n_fft=2048
            )
            .transpose()
            .astype(np.float32)
        )

        return sample, self.labels[idx]

    def __path(self, *args: str) -> str:
        return os.path.join(self.path, self.FOLDER, *args)

    @functools.lru_cache
    def available_files_by_class(self) -> SplitFiles:
        files: Dict[str, Set[str]] = {}

        for _class in self.CLASSES:
            directory = self.__path(_class)
            files[_class] = set(os.listdir(directory))

        return files

    def files_by_class(
        self, split: Literal["train", "val", "test"]
    ) -> SplitFiles:
        if split in ("val", "test"):
            return self.__files_by_class(split)

        test_files = self.__files_by_class("test")
        val_files = self.__files_by_class("val")

        files = self.available_files_by_class()
        train_files = {}

        for _class in self.CLASSES:
            files[_class] -= test_files[_class]
            files[_class] -= val_files[_class]

            train_files[_class] = files[_class]

        return train_files

    def __files_by_class(self, split: Literal["val", "test"]) -> SplitFiles:
        split_filename = self.VAL_LIST if split == "val" else self.TEST_LIST
        filepath = self.__path(split_filename)

        with open(filepath) as f:
            files = f.readlines()

        split_files: Dict[str, Set[str]] = {}
        available_files = self.available_files_by_class()

        for file in files:
            match = self.FILEPATH_RE.match(file)

            if match is None:
                continue

            _class = match.group("class")
            filename = match.group("file")

            if (
                _class not in self.CLASSES
                or filename not in available_files[_class]
            ):
                continue

            if _class not in split_files:
                split_files[_class] = set()

            split_files[_class].add(filename)

        return split_files
