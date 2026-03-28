# Tarefa 3: Integração Industrial e WordPiece
#
# O BERT usa o algoritmo WordPiece, que tem uma lógica probabilística
# parecida com o BPE mas com algumas diferenças na forma de escolher as fusões.
# Aqui usamos o tokenizador multilíngue já treinado via Hugging Face.

from transformers import AutoTokenizer


def run_wordpiece():
    # Carrega o tokenizador do BERT multilíngue e aplica na frase de teste,
    # exibindo como o modelo divide as palavras em sub-palavras
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
    tokens = tokenizer.tokenize(frase)

    print(f"Frase original:\n{frase}")
    print(f"\nTokens WordPiece:\n{tokens}")
