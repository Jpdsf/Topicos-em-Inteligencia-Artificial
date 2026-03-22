import torch
from lab_05.training import Trainer

# batch minúsculo para o teste rodar rápido
SRC = [[101, 1037, 2158, 102, 0], [101, 1037, 3899, 102, 0]]
TGT_IN = [[101, 4371, 102, 0, 0], [101, 4371, 9587, 102, 0]]
TGT_OUT = [[4371, 102, 0, 0, 0], [4371, 9587, 102, 0, 0]]


def test_trainer_instantiates():
    trainer = Trainer()
    assert trainer.model is not None
    assert trainer.criterion is not None
    assert trainer.optimizer is not None


def test_loss_decreases():
    trainer = Trainer()
    history = trainer.train(SRC, TGT_IN, TGT_OUT)

    assert len(history) > 0
    # o loss da última época deve ser menor que o da primeira
    assert history[-1] < history[0], (
        f"Loss não caiu: início={history[0]:.4f}, fim={history[-1]:.4f}"
    )


def test_output_shape():
    trainer = Trainer()
    src = torch.tensor(SRC, dtype=torch.long).to(trainer.device)
    tgt = torch.tensor(TGT_IN, dtype=torch.long).to(trainer.device)

    trainer.model.eval()
    with torch.no_grad():
        output = trainer.model(src, tgt)

    batch, seq_len, vocab = output.shape
    assert batch == 2
    assert vocab == 30522
