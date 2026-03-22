from lab_05.data import load_translation_subset


def test_load_returns_correct_count():
    pairs = load_translation_subset(num_samples=10)
    assert len(pairs) == 10


def test_load_has_src_and_tgt_keys():
    pairs = load_translation_subset(num_samples=5)
    for pair in pairs:
        assert "src" in pair
        assert "tgt" in pair


def test_load_values_are_strings():
    pairs = load_translation_subset(num_samples=5)
    for pair in pairs:
        assert isinstance(pair["src"], str)
        assert isinstance(pair["tgt"], str)
