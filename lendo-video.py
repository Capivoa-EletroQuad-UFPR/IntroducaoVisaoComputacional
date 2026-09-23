import argparse
from pathlib import Path
import cv2

# configuração dos argumentos de linha de comando
parser = argparse.ArgumentParser(description="Exibir vídeo sobrepondo texto.")
parser.add_argument(
    "-v", "--video", 
    type=str, 
    default="0", 
    help="Caminho do arquivo de vídeo ou índice da webcam (padrão: 0)"
)
parser.add_argument(
    "-t", "--texto", 
    type=str, 
    default="Visao Computacional - Capivoa", 
    help="Texto a ser desenhado no frame"
)
parser.add_argument(
    "-o", "--output-dir",
    type=str,
    default="frames-capturados",
    help="Diretório onde os frames salvos serão armazenados"
)
args = parser.parse_args()

# cria o diretório se não existir
output_path = Path(args.output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# se o argumento for puramente numérico, converte para int (índice da câmera)
video_path = int(args.video) if args.video.isdigit() else args.video
texto = args.texto

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir a captura de vídeo.")
    exit()

index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou falha ao receber o frame.")
        break

    # parâmetros do texto
    posicao = (30, 50)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.9
    cor = (0, 255, 0)
    espessura = 2
    tipo_linha = cv2.LINE_AA

    cv2.putText(
        img=frame,
        text=texto,
        org=posicao,
        fontFace=fonte,
        fontScale=escala,
        color=cor,
        thickness=espessura,
        lineType=tipo_linha
    )

    cv2.imshow("Stream de Video com Texto", frame)

    key = cv2.waitKey(1) & 0xFF
    # sair do loop ao pressionar 'q'
    if key == ord('q'):
        break

    # salvar o frame ao pressionar 's'
    elif key == ord('s'):
        nome_arquivo = output_path / f"video-frame-{index:03d}.jpg"
        cv2.imwrite(str(nome_arquivo), frame)
        print(f"Frame salvo com sucesso: {nome_arquivo}")

    index += 1

cap.release()
cv2.destroyAllWindows()
