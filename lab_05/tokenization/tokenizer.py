from transformers import AutoTokenizer
from lab_05.config import (
    TOKENIZER_MODEL,
    TOKENIZER_MAX_LENGTH,
    TOKENIZER_PAD_ID,
)


class Tokenizer:

    def __init__(self):
        self._tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

        # tokens especiais
        self.pad_id = TOKENIZER_PAD_ID
        self.start_id = self._tok.cls_token_id   # [CLS] age como <START>
        self.eos_id = self._tok.sep_token_id      # [SEP] age como <EOS>

    def encode(self, text: str, max_length: int = TOKENIZER_MAX_LENGTH) -> list[int]:
        return self._tok.encode(
            text,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

    def encode_src(self, text: str) -> list[int]:
        return self.encode(text)

    def encode_tgt(self, text: str) -> list[int]:

        ids = self.encode(text)
        decoder_input = [self.start_id] + ids
        decoder_target = ids + [self.eos_id]
        return decoder_input, decoder_target

    def pad(self, sequences: list[list[int]]) -> list[list[int]]:
        max_len = max(len(s) for s in sequences)
        return [s + [self.pad_id] * (max_len - len(s)) for s in sequences]

    def encode_batch(
        self, pairs: list[dict]
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:

        src_list, tgt_in_list, tgt_out_list = [], [], []

        for pair in pairs:
            src_list.append(self.encode_src(pair["src"]))
            tgt_in, tgt_out = self.encode_tgt(pair["tgt"])
            tgt_in_list.append(tgt_in)
            tgt_out_list.append(tgt_out)

        return (
            self.pad(src_list),
            self.pad(tgt_in_list),
            self.pad(tgt_out_list),
        )
