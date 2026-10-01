# Introdução à Visão Computacional

Repositório complementar de suporte às aulas de Introdução à Visão Computacional (SAEM).

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
├── aula-2/                       # Conteúdo prático do Módulo 2
│   ├── criar_imagem.py           # Gerador da imagem de teste e templates sintéticos
│   ├── limiarizacao.py           # Fixo, Adaptativo (Média/Gauss), Otsu e Kittler-Illingworth
│   ├── bordas_sobel_canny.py     # Detecção de bordas: Sobel (Gx, Gy, Mag) vs Canny (1px)
│   ├── contornos_momentos.py     # Contornos, hierarquias (furos) e centróides via momentos
│   ├── template_matching.py      # Busca por molde (SQDIFF, CCORR, CCOEFF) e limitações
│   ├── yolo_deteccao.py          # Conceito YOLO: Grid SxS, propostas e limpeza por NMS
│   ├── yolo_generico.py          # Detector e classificador genérico em tempo real (Ultralytics)
│   └── imagem_teste_aula2.png    # Imagem de teste com iluminação irregular e peças
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação geral
```

---

## Mapeamento dos Scripts

### Módulo 1 (Aula 1)

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

### Módulo 2 (Aula 2)

|                   Tópico                    | Script em `aula-2/`                                     | Conceitos e Operações Abordadas                                                                                                                                                                            |
| :-----------------------------------------: | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|         **0. Preparação de Dados**          | [`criar_imagem.py`](aula-2/criar_imagem.py)             | Gera `imagem_teste_aula2.png` com gradiente de luz não-uniforme, objetos vazados (furos), engrenagens, chips e templates (`template_porca.png`, `template_engrenagem.png`, `template_chip.png`).           |
|       **1. Métodos de Limiarização**        | [`limiarizacao.py`](aula-2/limiarizacao.py)             | Segmentação de planos: Limiar Fixo (`cv2.threshold`), Adaptativo por Média e Gaussiana (`cv2.adaptiveThreshold`), Método de Otsu (`THRESH_OTSU`) e algoritmo de Kittler-Illingworth (Minimum Error Bayes). |
|  **2. Detecção de Bordas: Sobel e Canny**   | [`bordas_sobel_canny.py`](aula-2/bordas_sobel_canny.py) | Fronteiras visuais, kernels Sobel $G_x$ e $G_y$, magnitude do gradiente vs Algoritmo de Canny de 4 etapas (Gauss, gradiente, supressão de não-máximos e histerese para bordas de 1 pixel).                 |
| **3. Contornos, Hierarquias e Centróides**  | [`contornos_momentos.py`](aula-2/contornos_momentos.py) | Representação vetorial com `cv2.findContours` (`RETR_TREE` vs `RETR_EXTERNAL`), detecção de furos internos, descarte de ruído por área e cálculo de centróide $(c_x, c_y)$ via momentos espaciais.         |
|    **4. Detecção com Template Matching**    | [`template_matching.py`](aula-2/template_matching.py)   | Busca por molde de referência com `cv2.matchTemplate`, comparação dos métodos (`TM_SQDIFF_NORMED`, `TM_CCORR_NORMED`, `TM_CCOEFF_NORMED`) e demonstração das limitações de rotação, escala e oclusão.      |
|     **5. Conceito e Pipeline do YOLO**      | [`yolo_deteccao.py`](aula-2/yolo_deteccao.py)           | Visão profunda em passagem única: divisão em Grid celular $S \times S$, matching boxes com probabilidade de classes e limpeza de duplicatas via Non-Maximum Suppression (`cv2.dnn.NMSBoxes`).              |
| **6. Detector Genérico YOLO em Tempo Real** | [`yolo_generico.py`](aula-2/yolo_generico.py)           | Inferência genérica com rede YOLO real (`ultralytics`), com detecção multi-classe, bounding boxes e centróides em fotos, vídeos ou webcam.                                                                 |
|          **7. Desafio da Aula 2**           | [`desafio/desafio.py`](aula-2/desafio/desafio.py)       | Pipeline integrado: carrega frames da esteira, segmenta com limiarização, rastreia contornos, descarta ruídos, calcula centróides, gera caixas delimitadoras e valida classes via Template Matching.       |

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

---

### Execução dos Scripts do Módulo 2 (Aula 2):

#### 1. Gerar a Imagem Didática e Templates

Gera a cena sintética com gradiente de iluminação, porcas vazadas, engrenagens e salva os moldes de referência:

```bash
python aula-2/criar_imagem.py
```

#### 2. Métodos de Limiarização e Segmentação de Planos

Compara Limiar Fixo, Adaptativo (Média/Gaussiano), Otsu e Kittler-Illingworth:

- **`1` a `5`**: Alterna entre os métodos
- **`TAB`**: Exibe painel comparativo com todos os métodos lado a lado
- **`+` / `-`**: Ajusta o limiar $T$ em tempo real
- **`I`**: Inverte a máscara binária

```bash
python aula-2/limiarizacao.py
python aula-2/limiarizacao.py --metodo otsu
```

#### 3. Detecção de Bordas: Sobel vs Canny

Compara o operador matricial de Sobel com as 4 etapas do algoritmo de Canny:

- **`TAB`**: Alterna entre visão comparativa e decomposição ($G_x, G_y, |G|$, Canny)
- **`W` / `S`**: Ajusta o limiar inferior de Canny
- **`E` / `D`**: Ajusta o limiar superior de Canny
- **`K`**: Alterna tamanho do kernel de Sobel (3, 5 ou 7)

```bash
python aula-2/bordas_sobel_canny.py
```

#### 4. Contornos, Hierarquias e Centróides

Extrai polígonos fechados, detecta furos internos e calcula centroides com momentos espaciais:

- **`H`**: Alterna hierarquia (`RETR_TREE` com furos vs `RETR_EXTERNAL`)
- **`B`**: Ativa/desativa caixas delimitadoras
- **`+` / `-`**: Ajusta o corte de ruído por área mínima

```bash
python aula-2/contornos_momentos.py --area-min 100
```

#### 5. Detecção de Objetos com Template Matching

Localiza objetos por correlação espacial, gera mapas de calor e demonstra limitações práticas:

- **`M`**: Alterna método (`TM_CCOEFF_NORMED`, `TM_SQDIFF_NORMED`, `TM_CCORR_NORMED`)
- **`T`**: Alterna molde de referência (Porca, Engrenagem, Chip)
- **`R`**: Rotaciona o template para evidenciar a sensibilidade a giro
- **`W` / `S`**: Altera a escala do template para testar sensibilidade dimensional
- **`0`**: Restaura rotação e escala originais

```bash
python aula-2/template_matching.py
```

#### 6. Conceito e Pipeline do YOLO

Demonstra visualmente a divisão em Grid celular $S \times S$, propostas brutas e limpeza via Non-Maximum Suppression (NMS):

- **`G`**: Ativa/desativa a grade celular
- **`+` / `-`**: Ajusta o limiar de confiança
- **`I`**: Ajusta a tolerância de sobreposição IoU

```bash
python aula-2/yolo_deteccao.py
```

#### 7. Detector Genérico YOLO em Tempo Real

Detecta e classifica dezenas de classes em imagem, vídeo ou webcam usando rede neural YOLO:

- **`ESPAÇO`**: Pausar / Continuar
- **`+` / `-`**: Ajustar limiar de confiança
- **`S`**: Salvar frame anotado

```bash
python aula-2/yolo_generico.py --midia video.mp4
python aula-2/yolo_generico.py --midia 0  # Webcam
python aula-2/yolo_generico.py --conf 0.60
```

#### 8. Solução do Desafio da Aula 2

Executa o rastreamento completo na esteira industrial: retificação, limiarização, contornos, centróides e validação com Template Matching:

- **`ESPAÇO`**: Pausar / Continuar o vídeo
- **`L`**: Alternar método de limiarização
- **`T`**: Ativar / desativar validação por Template Matching
- **`+` / `-`**: Ajustar corte de área mínima

```bash
python aula-2/desafio/desafio.py
python aula-2/desafio/desafio.py --midia video.mp4
```

---

## Slides

Acesse os [slides](https://canva.link/7h99gdonnvfqsd5) para ver os tópicos e desafios.
