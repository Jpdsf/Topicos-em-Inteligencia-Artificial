import torch
from lab_05.transformer import Transformer
from lab_05.tokenization import Tokenizer
from lab_05.config import (
    MODEL_VOCAB_SIZE,
    MODEL_D_MODEL,
    MODEL_NUM_HEADS,
    MODEL_NUM_LAYERS,
    MODEL_D_FF,
    TOKENIZER_PAD_ID,
    OVERFIT_EPOCHS,
    OVERFIT_LR,
    OVERFIT_MAX_NEW_TOKENS,
)
import torch.nn as nn
from torch.optim import Adam


def overfit_test(src_sentence: str, tgt_sentence: str) -> str:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer()

    src_ids = tok.encode_src(src_sentence)
    tgt_in_ids, tgt_out_ids = tok.encode_tgt(tgt_sentence)

    src = torch.tensor([src_ids],    dtype=torch.long).to(device)
    tgt_in = torch.tensor([tgt_in_ids], dtype=torch.long).to(device)
    tgt_out = torch.tensor([tgt_out_ids], dtype=torch.long).to(device)

    model = Transformer(
        vocab_size=MODEL_VOCAB_SIZE,
        d_model=MODEL_D_MODEL,
        num_heads=MODEL_NUM_HEADS,
        num_layers=MODEL_NUM_LAYERS,
        d_ff=MODEL_D_FF,
        dropout=0.0,  # sem dropout no overfitting test
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=TOKENIZER_PAD_ID)
    optimizer = Adam(model.parameters(), lr=OVERFIT_LR)

    print(f"\n[Overfit] Treinando em: '{src_sentence}' → '{tgt_sentence}'")
    model.train()
    for epoch in range(1, OVERFIT_EPOCHS + 1):
        optimizer.zero_grad()
        output = model(src, tgt_in)

        seq_len, vocab_size = output.shape[1], output.shape[2]
        loss = criterion(
            output.reshape(seq_len, vocab_size),
            tgt_out.reshape(seq_len),
        )
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4}/{OVERFIT_EPOCHS} | Loss: {loss.item():.4f}")

    translation = _greedy_decode(model, tok, src, device)
    print(f"\n[Overfit] Tradução gerada : '{translation}'")
    print(f"[Overfit] Tradução esperada: '{tgt_sentence}'")
    return translation


def _greedy_decode(
    model: Transformer,
    tok: Tokenizer,
    src: torch.Tensor,
    device: torch.device,
) -> str:
    model.eval()
    with torch.no_grad():
        generated = [tok.start_id]

        for _ in range(OVERFIT_MAX_NEW_TOKENS):
            tgt_in = torch.tensor([generated], dtype=torch.long).to(device)
            output = model(src, tgt_in)

            next_token = output[0, -1, :].argmax(dim=-1).item()
            generated.append(next_token)

            if next_token == tok.eos_id:
                break

    # decodifica ignorando START e EOS
    token_ids = [t for t in generated if t not in (tok.start_id, tok.eos_id)]
    return tok._tok.decode(token_ids, skip_special_tokens=True)
