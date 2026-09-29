"""
CAPIVOA UFPR.

Demonstração dos conceitos fundamentais de Visão Computacional
apresentados no Módulo 1 (Capivoa - EletroQuad-UFPR).

Tópicos abordados:
1. Estrutura da Imagem :
  - A imagem como matriz MxN de pixels no NumPy.
  - Atributos: shape (altura, largura, canais), dtype (uint8: 0 a 255).
  - Acesso e inspeção do valor numérico de um pixel específico.
2. Canais de Cor BGR vs RGB :
  - Separação dos canais Azul (B), Verde (G) e Vermelho (R) com cv2.split.
  - Entendendo o porquê do padrão BGR no OpenCV e conversão cv2.COLOR_BGR2RGB.
3. Espaços de Cores :
  - Conversão para Escala de Cinza (Grayscale): cv2.COLOR_BGR2GRAY.
  - Conversão para HSV (Hue, Saturation, Value): cv2.COLOR_BGR2HSV.
4. Entrada e Saída Básica com OpenCV :
  - Leitura: cv2.imread
  - Exibição: cv2.imshow e controle de janela com cv2.waitKey / cv2.destroyAllWindows
  - Escrita: cv2.imwrite

Como executar:
  python comandos_basicos.py
  python comandos_basicos.py --midia imagem.jpg
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# permite importar o pacote 'shared' executando da raiz ou de dentro de aula-1/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from shared.media_io import carregar_imagem, criar_parser_midia, concatenar_lado_a_lado


def main():
  """
  Executa a demonstração dos comandos básicos e estrutura matricial de imagens.

  Carrega a imagem de teste, analisa propriedades da matriz NumPy (dimensões,
  tipo de dado, intensidades mínimas e máximas), decompõe os canais BGR,
  aplica conversões para Grayscale e HSV, grava a imagem processada em disco
  e exibe painéis comparativos didáticos.
  """
  parser = criar_parser_midia(
    descricao="Operações Básicas e Estrutura da Imagem ",
    nome="imagem_teste.png"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa o processamento sem abrir janelas gráficas."
  )
  args = parser.parse_args()

  print("=" * 60)
  print("CAPIVOA - INTRODUÇÃO À VISÃO COMPUTACIONAL (MÓDULO 1)")
  print("COMANDOS BÁSICOS & ESTRUTURA DA IMAGEM")
  print("=" * 60)

  # 1. leitura da imagem (cv2.imread)
  caminho_imagem = args.midia
  print(f"\n[1] Carregando imagem: '{caminho_imagem}'...")
  img_bgr = carregar_imagem(caminho_imagem)

  # 2. estrutura da matriz e quantização
  altura, largura, canais = img_bgr.shape
  total_pixels = altura * largura
  tipo_dado = img_bgr.dtype
  valor_min = img_bgr.min()
  valor_max = img_bgr.max()

  print("\n--- Estrutura Matricial ---")
  print(f"Dimensões (shape) : Altura={altura} px | Largura={largura} px | Canais={canais}")
  print(f"Total de pixels  : {total_pixels:,} pixels")
  print(f"Tipo de dado   : {tipo_dado} (uint8 -> 8 bits, faixa [0, 255])")
  print(f"Intensidades   : Mínima={valor_min} | Máxima={valor_max}")

  # acesso a um pixel pontual (linha y=70, coluna x=70 - região do bloco azul da imagem de teste)
  y_amostra, x_amostra = min(70, altura - 1), min(70, largura - 1)
  pixel_bgr = img_bgr[y_amostra, x_amostra]
  print(f"Valor do pixel em (x={x_amostra}, y={y_amostra}) [BGR]: {pixel_bgr.tolist()}")

  # 3. canais de cor (bgr)
  print("\n--- Separação de Canais ---")
  canal_b, canal_g, canal_r = cv2.split(img_bgr)
  print("Canais divididos com sucesso:")
  print(" - Canal B (Azul)  : valores mais altos onde predomina a cor azul")
  print(" - Canal G (Verde)  : valores mais altos onde predomina a cor verde")
  print(" - Canal R (Vermelho): valores mais altos onde predomina a cor vermelha")

  # 4. espaços de cores (grayscale e hsv)
  print("\n--- Espaços de Cores ---")
  # grayscale: 1 canal de intensidade (0=preto, 255=branco), fórmula: y = 0.299*r + 0.587*g + 0.114*b
  img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  print(f"Escala de Cinza: shape={img_cinza.shape} (1 único canal de intensidade)")

  # hsv: hue (matiz 0-179 no opencv), saturation (0-255), value/brilho (0-255)
  img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
  canal_h, canal_s, canal_v = cv2.split(img_hsv)
  print(f"HSV      : shape={img_hsv.shape} (Hue/Matiz, Saturation/Saturação, Value/Brilho)")

  # 5. escrita / salvamento em disco (cv2.imwrite)
  diretorio_saida = Path("frames-capturados")
  diretorio_saida.mkdir(parents=True, exist_ok=True)
  caminho_cinza = diretorio_saida / "exemplo_escala_cinza.png"
  cv2.imwrite(str(caminho_cinza), img_cinza)
  print(f"\n[5] Imagem em escala de cinza salva com cv2.imwrite em: '{caminho_cinza}'")

  # 6. exibição visual interativa (cv2.imshow e cv2.waitkey)
  # monta painéis comparativos didáticos lado a lado
  painel_canais = concatenar_lado_a_lado(
    [img_bgr, canal_b, canal_g, canal_r],
    titulos=["Original (BGR)", "Canal B (Azul)", "Canal G (Verde)", "Canal R (Vermelho)"],
    altura_padrao=320
  )

  painel_espacos = concatenar_lado_a_lado(
    [img_bgr, img_cinza, canal_h, canal_s],
    titulos=["Original (BGR)", "Grayscale (Cinza)", "HSV: Canal H (Matiz)", "HSV: Canal S (Saturacao)"],
    altura_padrao=320
  )

  if not args.no_show:
    janela1 = "Comandos Basicos - Painel 1: Canais BGR"
    janela2 = "Comandos Basicos - Painel 2: Espacos de Cores"

    print("\n[6] Exibindo janelas interativas com OpenCV (pressione 'q' ou ESC para fechar)...")
    cv2.imshow(janela1, painel_canais)
    cv2.imshow(janela2, painel_espacos)

    # loop de espera interativo com suporte a 'q', esc e fechamento no 'x'
    while True:
      tecla = cv2.waitKey(50) & 0xFF
      # encerra ao pressionar 'q' ou esc (código 27)
      if tecla in [ord("q"), ord("Q"), 27]:
        break
      # encerra se qualquer janela for fechada pelo botão 'x'
      if (
        cv2.getWindowProperty(janela1, cv2.WND_PROP_VISIBLE) < 1
        or cv2.getWindowProperty(janela2, cv2.WND_PROP_VISIBLE) < 1
      ):
        break

    cv2.destroyAllWindows()
    print("Janelas encerradas com sucesso.")
  else:
    print("\n[6] Modo no-show ativado: processamento e validação das matrizes concluídos.")


if __name__ == "__main__":
  main()
