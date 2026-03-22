from lab_05.tokenization import Tokenizer

SAMPLE_PAIRS = [
    {"src": "A man is walking.", "tgt": "Ein Mann geht."},
    {"src": "A dog runs fast.", "tgt": "Ein Hund läuft schnell."},
]


def test_encode_src_returns_list_of_ints():
    tok = Tokenizer()
    ids = tok.encode_src("A man is walking.")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_tgt_has_start_and_eos():
    tok = Tokenizer()
    dec_input, dec_target = tok.encode_tgt("Ein Mann geht.")
    assert dec_input[0] == tok.start_id
    assert dec_target[-1] == tok.eos_id


def test_encode_tgt_lengths_match():
    tok = Tokenizer()
    dec_input, dec_target = tok.encode_tgt("Ein Mann geht.")
    assert len(dec_input) == len(dec_target)


def test_pad_equalizes_lengths():
    tok = Tokenizer()
    padded = tok.pad([[1, 2, 3], [1, 2], [1]])
    lengths = [len(s) for s in padded]
    assert len(set(lengths)) == 1


def test_encode_batch_returns_equal_length_sequences():
    tok = Tokenizer()
    src, tgt_in, tgt_out = tok.encode_batch(SAMPLE_PAIRS)

    src_lens = set(len(s) for s in src)
    tgt_in_lens = set(len(s) for s in tgt_in)
    tgt_out_lens = set(len(s) for s in tgt_out)

    assert len(src_lens) == 1
    assert len(tgt_in_lens) == 1
    assert len(tgt_out_lens) == 1
