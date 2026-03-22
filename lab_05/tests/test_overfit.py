from lab_05.inference import overfit_test

SRC = "A man is walking."
TGT = "Ein Mann geht."


def test_overfit_returns_string():
    result = overfit_test(SRC, TGT)
    assert isinstance(result, str)
    assert len(result) > 0


def test_overfit_reproduces_target():
    result = overfit_test(SRC, TGT)
    assert result.strip() == TGT.strip(), (
        f"Esperado: '{TGT}' | Gerado: '{result}'"
    )
