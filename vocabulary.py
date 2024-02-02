import logging

from config import cfg


logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self):
        self._start_token = cfg.common.vocabulary.start_token
        self._end_token = cfg.common.vocabulary.end_token
        self._unknown_token = cfg.common.vocabulary.unknown_token
        self._pad_token = cfg.common.vocabulary.pad_token
        self._vehicle_tokens = cfg.common.vocabulary.vehicle_tokens
        self._pedestrian_tokens = cfg.common.vocabulary.pedestrian_tokens
        self._tokens = cfg.common.vocabulary.tokens

        maxWH = max(cfg.common.img_width_resize, cfg.common.img_height_resize)

        self._vocabulary = [
            self._start_token,
            self._end_token,
            self._unknown_token,
        ] + self._tokens

        # Define number tokens for bbox range 1:max(with,height)
        number_tokens = [str(num) for num in range(1, maxWH + 1)]

        # Add <pad> to the vocabulary with index -100
        self._vocabulary = [self._pad_token] + self._vocabulary + number_tokens

        self._vocab_size = len(self._vocabulary)

        # Assign an index to each word in the vocabulary
        self._word_to_idx = {word: idx for idx, word in enumerate(self._vocabulary)}
        self._idx_to_word = {idx: word for idx, word in enumerate(self._vocabulary)}

        # Set the index for <pad> as -100
        self._word_to_idx[self._pad_token] = -100

        logger.debug(f"VOCABULARY: {self._vocabulary}")

    @property
    def start_token(self):
        return self._start_token

    @property
    def end_token(self):
        return self._end_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def vehicle_tokens(self):
        return self._vehicle_tokens

    @property
    def pedestrian_tokens(self):
        return self._pedestrian_tokens

    @property
    def vocabulary(self):
        return self._vocabulary

    def get_word_from_index(self, index):
        return self._idx_to_word.get(index, self._unknown_token)

    def get_index_from_word(self, word):
        return self._word_to_idx.get(word, self._word_to_idx[self._unknown_token])

    def __len__(self):
        return len(self._vocabulary)
