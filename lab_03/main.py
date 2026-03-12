import sys
import os
from demos.demo_mask           import run as demo_mask
from demos.demo_cross_attention import run as demo_cross_attention
from demos.demo_inference       import run as demo_inference

from tests.test_mask      import run_all as test_mask
from tests.test_attention import run_all as test_attention
from tests.test_inference import run_all as test_inference

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("  LABORATÓRIO 3 — IMPLEMENTANDO O DECODER")


    demo_mask()
    demo_cross_attention()
    demo_inference()

    print("  SUÍTE DE TESTES")

    test_mask()
    test_attention()
    test_inference()

    print("  Laboratório 3 concluído com sucesso!")


if __name__ == "__main__":
    main()
