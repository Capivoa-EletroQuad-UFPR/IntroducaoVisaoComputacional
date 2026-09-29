# Introdução à Visão Computacional

Repositório complementar de suporte didático às aulas de Introdução à Visão Computacional (**Capivoa — EletroQuad-UFPR**).

O repositório é organizado em pastas modulares:

- **`shared/`**: Códigos e utilitários reutilizáveis por outras aulas e módulos (ex: [`media_io.py`](shared/media_io.py)).
- **`aula-1/`**: Códigos práticos, mídias e exercícios didáticos do **Módulo 1**.
- **`aula-2/`**: Espaço reservado para os tópicos do **Módulo 2**.

---

## Estrutura do Repositório

```text
├── shared/                       # Módulos compartilhados entre aulas
│   ├── __init__.py
│   └── media_io.py               # Leitura/escrita padronizada de mídia e concatenação visual
├── aula-1/                       # Conteúdo prático do Módulo 1
│   ├── criar_imagem.py           # Gerador da imagem de teste
│   ├── comandos_basicos.py       # Estrutura matricial, canais BGR/RGB e espaços de cores
│   ├── lendo_video.py            # Stream de vídeo/webcam com FPS e captura de frames
│   ├── corte_redimensionamento.py# Recorte (ROI) e métodos de interpolação
│   ├── translacao.py             # Translação 2x3 afim com controle interativo
│   ├── rotacao.py                # Rotação afim com exibição de pivô e controle de ângulo/escala
│   ├── trans_perspectiva.py      # Homografia 3x3 e retificação com 4 vértices
│   ├── filtros.py                # Convolução espacial (Média, Gaussiano, Mediana, Bilateral)
│   ├── stream_exemplo.py         # Exemplo polimórfico conciso com stream_midia
│   ├── imagem_teste.png          # Imagem didática para testes
│   └── video.mp4                 # Vídeo de teste para processamento contínuo
├── aula-2/                       # Conteúdo do Módulo 2 (próxima aula)
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação geral
```

---

## Mapeamento dos Scripts

|                      Tópico                      | Script em `aula-1/`                                               | Conceitos e Operações Abordadas                                                                                                                                                                         |
| :----------------------------------------------: | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|            **0. Preparação de Dados**            | [`criar_imagem.py`](aula-1/criar_imagem.py)                       | Gera a imagem didática `imagem_teste.png` contendo elementos sintéticos projetados para testar canais, geometria, ruídos e interpolações.                                                               |
|        **1. Estrutura da Imagem e Cores**        | [`comandos_basicos.py`](aula-1/comandos_basicos.py)               | Estrutura matricial (`shape`, `dtype`), quantização `uint8`, canais de cores BGR vs RGB (`cv2.split`), espaços de cores (Grayscale, HSV) e I/O com `cv2.imread`, `cv2.imshow`, `cv2.imwrite`.           |
|     **2. Operações Básicas: Vídeo e Câmera**     | [`lendo_video.py`](aula-1/lendo_video.py)                         | Leitura de vídeo e webcam com `cv2.VideoCapture`, medição de FPS real, exibição de HUD overlay e controles de pausa (`ESPAÇO`) e salvamento de frames (`S`).                                            |
|   **3. Transformações: Corte e Interpolação**    | [`corte_redimensionamento.py`](aula-1/corte_redimensionamento.py) | Fatiamento de Região de Interesse (`ROI = img[y1:y2, x1:x2]`), subamostragem e comparação lado a lado dos 4 métodos de interpolação (`INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`, `INTER_LANCZOS4`). |
|   **4. Transformações: Translação Euclidiana**   | [`translacao.py`](aula-1/translacao.py)                           | Transformação afim euclidiana 2x3 para translação $(t_x, t_y)$ via `cv2.warpAffine`. Inclui movimentação teclado.                                                                                       |
|     **5. Transformações: Rotação e Escala**      | [`rotacao.py`](aula-1/rotacao.py)                                 | Matriz afim 2x3 com `cv2.getRotationMatrix2D(centro, angulo, escala)`. Exibe o ponto pivô graficamente e permite girar e escalar interativamente pelas teclas.                                          |
| **6. Transformações: Perspectiva e Retificação** | [`trans_perspectiva.py`](aula-1/trans_perspectiva.py)             | Homografia projetiva 3x3 com `cv2.getPerspectiveTransform` e `cv2.warpPerspective`. Destaca os 4 vértices coloridos e retifica planos inclinados da cena.                                               |
|       **7. Filtros e Convolução Espacial**       | [`filtros.py`](aula-1/filtros.py)                                 | Convolução espacial e comparação dos 4 filtros de suavização: Média (`cv2.blur`), Gaussiano (`cv2.GaussianBlur`), Mediana (`cv2.medianBlur`) e Bilateral (`cv2.bilateralFilter`).                       |
|     **8. Fluxo de Processamento (Pipeline)**     | [`stream_exemplo.py`](aula-1/stream_exemplo.py)                   | Exemplo didático conciso demonstrando processamento polimórfico (o mesmo pipeline executa de forma transparente em imagem estática, vídeo ou webcam).                                                   |

---

## Instalação e Configuração

### 1. Criar e Ativar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux / macOS
# ou no Windows: venv\Scripts\activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## Execução dos Scripts

Os scripts podem ser executados tanto da **raiz do repositório** quanto de dentro da pasta **`aula-1/`**.
Todos aceitam o parâmetro `--midia` / `-m` ou `--origem` / `-o`:

- **Imagem de Teste Padrão:** omitir o parâmetro ou usar `-m imagem_teste.png`
- **Arquivo de Vídeo:** `-m video.mp4`
- **Webcam ao Vivo:** `-m 0` (índice da câmera)

---

### Passo a Passo da Execução:

#### 1. Gerar a Imagem Didática de Teste

Gera a imagem base com elementos geométricos, canais puros e ruídos sintéticos:

```bash
python aula-1/criar_imagem.py
```

#### 2. Comandos Básicos e Estrutura Matricial

Exibe informações de dimensões (`shape`), tipo de dado (`uint8`), separação de canais BGR e conversão para escala de cinza e HSV:

```bash
python aula-1/comandos_basicos.py
python aula-1/comandos_basicos.py --midia imagem.jpg
```

#### 3. Leitura e Monitoramento de Vídeo / Câmera

Reproduz vídeo ou stream contínuo exibindo telemetria em tempo real (FPS medido, número do frame, resolução):

- **`ESPAÇO`**: Pausar / Retomar a reprodução
- **`S`**: Salvar frame atual em `frames-capturados/`
- **`Q`**: Encerrar

```bash
python aula-1/lendo_video.py --midia video.mp4
python aula-1/lendo_video.py --midia 0  # Webcam ao vivo
```

#### 4. Corte (ROI) e Métodos de Interpolação

Extrai a região de interesse e compara os métodos `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC` e `INTER_LANCZOS4` em processo de subamostragem e ampliação:

```bash
python aula-1/corte_redimensionamento.py
```

#### 5. Translação Euclidiana Interativa

Desloque a imagem em tempo real por meio da matriz afim:

- Pressione **`A` / `D`** para mover horizontalmente ($-/+ t_x$).
- Pressione **`W` / `S`** para mover verticalmente ($-/+ t_y$).
- Pressione **`R`** para resetar para a posição original.

```bash
python aula-1/translacao.py
python aula-1/translacao.py --midia video.mp4
```

#### 6. Rotação com Pivô e Escala

Gire a imagem em torno do ponto pivô central e ajuste a escala interativamente:

- Pressione **`A` / `D`** para rotacionar anti-horário / horário.
- Pressione **`W` / `S`** para aumentar / diminuir a escala (zoom).
- Pressione **`R`** para resetar ângulo e escala.

```bash
python aula-1/rotacao.py --angulo 45
python aula-1/rotacao.py --midia video.mp4
```

#### 7. Transformação de Perspectiva e Retificação

Identifica 4 vértices correspondentes e retifica o plano inclinado em vista ortogonal frontal:

```bash
python aula-1/trans_perspectiva.py
python aula-1/trans_perspectiva.py --midia video.mp4
```

#### 8. Filtros e Convolução Espacial

Compara a remoção de ruídos (Sal e Pimenta vs. Gaussiano) pelos 4 filtros clássicos:

```bash
python aula-1/filtros.py --kernel 5
```

#### 9. Exemplo de Processamento Polimórfico

Demonstra como o mesmo loop de algoritmo funciona de forma transparente em imagens, vídeos ou câmeras:

```bash
python aula-1/stream_exemplo.py --midia imagem_teste.png
python aula-1/stream_exemplo.py --midia video.mp4
```
