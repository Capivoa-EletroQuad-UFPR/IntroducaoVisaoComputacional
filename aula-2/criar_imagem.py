"""
CAPIVOA UFPR.

Gera uma imagem de teste sintética ('imagem_teste_aula2.png') e templates de referência
utilizados pelos scripts práticos do Módulo 2 de Introdução à Visão Computacional
(Capivoa - EletroQuad-UFPR).

A imagem foi projetada para demonstrar e validar os seguintes conceitos:
1. Limiarização (Fixo, Adaptativo, Otsu e Kittler-Illingworth):
   - Fundo com gradiente de iluminação não-uniforme (efeito de sombra gradual).
   - Objetos de alto e médio contraste para evidenciar a falha do limiar global fixo
     e a eficácia dos métodos adaptativos e automáticos.
2. Detecção de Bordas (Sobel e Canny):
   - Formas com bordas nítidas, diagonais, arcos e variações de intensidade.
3. Contornos, Hierarquias e Momentos:
   - Formas com furos internos (porcas, arruelas, engrenagens) para demonstrar
     a hierarquia pai/filho (cv2.RETR_TREE vs cv2.RETR_EXTERNAL).
   - Manchas espúrias de ruído de pequena área para testar filtro por área mínima.
4. Detecção de Objetos (Template Matching e YOLO):
   - Objetos repetidos e isolados para busca por molde (parafuso, engrenagem, chip).

Como executar:
  python criar_imagem.py            # Gera as imagens e abre visualização gráfica
  python criar_imagem.py --no-show  # Apenas gera e salva os arquivos PNG no disco
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))


def desenhar_porca_sextavada(img: np.ndarray, centro: tuple, raio_ext: int, raio_furo: int, cor: tuple):
  """
  Desenha uma porca sextavada com furo central vazado.

  Parametros
  ----------
  img : np.ndarray
    Imagem de destino.
  centro : tuple
    Coordenadas (x, y) do centro.
  raio_ext : int
    Raio externo do hexágono.
  raio_furo : int
    Raio do furo central.
  cor : tuple
    Cor BGR de preenchimento.
  """
  cx, cy = centro
  angulos = np.linspace(0, 2 * np.pi, 7)[:-1]
  pts = np.array([
    [int(cx + raio_ext * np.cos(a)), int(cy + raio_ext * np.sin(a))]
    for a in angulos
  ], dtype=np.int32)

  cv2.fillPoly(img, [pts], cor)
  # furo central com a cor aproximada do fundo local
  cor_furo = int(img[cy, cx, 0]) if img.ndim == 3 else int(img[cy, cx])
  cv2.circle(img, (cx, cy), raio_furo, (cor_furo, cor_furo, cor_furo), -1)
  cv2.circle(img, (cx, cy), raio_furo, (30, 30, 30), 1, lineType=cv2.LINE_AA)
  cv2.polylines(img, [pts], isClosed=True, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)


def desenhar_engrenagem(img: np.ndarray, centro: tuple, raio_corpo: int, raio_furo: int, num_dentes: int, cor: tuple):
  """
  Desenha uma engrenagem circular dentada com furo central.

  Parametros
  ----------
  img : np.ndarray
    Imagem de destino.
  centro : tuple
    Coordenadas (x, y) do centro.
  raio_corpo : int
    Raio do corpo principal da engrenagem.
  raio_furo : int
    Raio do furo central.
  num_dentes : int
    Quantidade de dentes externos.
  cor : tuple
    Cor BGR.
  """
  cx, cy = centro
  # dentes da engrenagem
  pontos = []
  passo = 2 * np.pi / (num_dentes * 2)
  for i in range(num_dentes * 2):
    r = raio_corpo + (8 if i % 2 == 1 else 0)
    ang = i * passo
    pontos.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])

  pts = np.array(pontos, dtype=np.int32)
  cv2.fillPoly(img, [pts], cor)
  cor_furo = int(img[cy, cx, 0]) if img.ndim == 3 else int(img[cy, cx])
  cv2.circle(img, (cx, cy), raio_furo, (cor_furo, cor_furo, cor_furo), -1)
  cv2.circle(img, (cx, cy), raio_furo, (25, 25, 25), 1, lineType=cv2.LINE_AA)
  cv2.polylines(img, [pts], isClosed=True, color=(25, 25, 25), thickness=2, lineType=cv2.LINE_AA)


def desenhar_chip_eletronico(img: np.ndarray, top_left: tuple, largura: int, altura: int, cor_corpo: tuple):
  """
  Desenha um circuito integrado (chip) retangular com pinos metálicos nas laterais.
  """
  x, y = top_left
  # pinos metálicos esquerdos e direitos
  num_pinos = 5
  espaco_y = altura // (num_pinos + 1)
  for i in range(1, num_pinos + 1):
    py = y + i * espaco_y
    # pino esquerdo
    cv2.rectangle(img, (x - 10, py - 3), (x, py + 3), (210, 210, 210), -1)
    # pino direito
    cv2.rectangle(img, (x + largura, py - 3), (x + largura + 10, py + 3), (210, 210, 210), -1)

  # corpo do chip
  cv2.rectangle(img, (x, y), (x + largura, y + altura), cor_corpo, -1)
  cv2.rectangle(img, (x, y), (x + largura, y + altura), (20, 20, 20), 2)
  # chanfro/marca indicadora do pino 1
  cv2.circle(img, (x + 12, y + 12), 4, (120, 120, 120), -1)
  cv2.putText(img, "IC-01", (x + 14, y + altura // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def gerar_imagem_aula2(caminho_saida: Path) -> np.ndarray:
  """
  Gera a cena sintética principal e salva em disco.

  Parametros
  ----------
  caminho_saida : Path
    Caminho onde a imagem será salva.

  Returns
  -------
  np.ndarray
    Matriz BGR da imagem gerada.
  """
  largura, altura = 800, 520

  # 1. fundo base com gradiente de iluminação suave (da esquerda mais iluminada para a direita mais escura)
  # isso cria o cenário perfeito para comparar limiar fixo vs adaptativo vs Otsu
  x_coords = np.linspace(230, 90, largura, dtype=np.float32)
  fundo_canal = np.tile(x_coords, (altura, 1))

  # adiciona uma vinheta diagonal sutil
  y_coords = np.linspace(15, -15, altura, dtype=np.float32)[:, None]
  fundo_canal = np.clip(fundo_canal + y_coords, 40, 250).astype(np.uint8)
  img = cv2.merge([fundo_canal, fundo_canal, fundo_canal])

  # 2. desenha as peças principais
  # Porcas sextavadas com furos internos
  desenhar_porca_sextavada(img, (140, 150), raio_ext=42, raio_furo=16, cor=(45, 55, 60))
  desenhar_porca_sextavada(img, (650, 170), raio_ext=40, raio_furo=15, cor=(35, 45, 50))
  desenhar_porca_sextavada(img, (400, 390), raio_ext=34, raio_furo=13, cor=(40, 50, 55))

  # Engrenagens circulares dentadas
  desenhar_engrenagem(img, (380, 160), raio_corpo=46, raio_furo=18, num_dentes=10, cor=(50, 60, 75))
  desenhar_engrenagem(img, (150, 380), raio_corpo=50, raio_furo=20, num_dentes=12, cor=(45, 55, 70))

  # Chips eletrônicos
  desenhar_chip_eletronico(img, (600, 340), largura=90, altura=60, cor_corpo=(25, 25, 30))

  # 3. Pequenas manchas de ruído espúrio (para exercitar filtro de contornos por área)
  np.random.seed(42)
  for _ in range(25):
    rx = np.random.randint(20, largura - 20)
    ry = np.random.randint(20, altura - 20)
    raio = np.random.randint(1, 3)
    cv2.circle(img, (rx, ry), raio, (20, 20, 20), -1)

  # salva a imagem principal
  caminho_saida.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(caminho_saida), img)
  print(f"[OK] Imagem didática da Aula 2 salva em: '{caminho_saida}'")

  # 4. gera e salva os templates isolados para Template Matching
  pasta_aula2 = caminho_saida.parent

  # Template da porca sextavada (recorte limpo do objeto em (140, 150))
  template_porca = img[150 - 46:150 + 46, 140 - 46:140 + 46].copy()
  cv2.imwrite(str(pasta_aula2 / "template_porca.png"), template_porca)

  # Template da engrenagem (recorte do objeto em (380, 160))
  template_engrenagem = img[160 - 58:160 + 58, 380 - 58:380 + 58].copy()
  cv2.imwrite(str(pasta_aula2 / "template_engrenagem.png"), template_engrenagem)

  # Template do chip
  template_chip = img[340 - 5:340 + 65, 600 - 15:600 + 105].copy()
  cv2.imwrite(str(pasta_aula2 / "template_chip.png"), template_chip)

  print(f"[OK] Templates salvos em '{pasta_aula2}': template_porca.png, template_engrenagem.png, template_chip.png")
  return img


def main():
  parser = argparse.ArgumentParser(description="Gera a imagem sintética didática da Aula 2 (Capivoa UFPR).")
  parser.add_argument(
    "-o", "--output",
    type=str,
    default="imagem_teste_aula2.png",
    help="Nome do arquivo de saída (padrão: imagem_teste_aula2.png)"
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Executa a geração sem abrir a janela de visualização."
  )
  args = parser.parse_args()

  caminho_saida = Path(__file__).resolve().parent / args.output
  img = gerar_imagem_aula2(caminho_saida)

  if not args.no_show:
    plt.figure(figsize=(10, 6.5))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("CAPIVOA UFPR - Aula 2: Imagem Sintética de Teste\n(Gradiente de Luz, Objetos Vazados e Ruídos)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
  main()
