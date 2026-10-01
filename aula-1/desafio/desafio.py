import sys
from pathlib import Path
import cv2
import numpy as np


def main():
    # carregar o vídeo no código
    caminho_video = str(Path(__file__).resolve().parent / "video.mp4")
    if len(sys.argv) > 1:
        caminho_video = sys.argv[1]
    cap = cv2.VideoCapture(caminho_video)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {caminho_video}")
        return

    # configuracao para salvar o video tratado
    caminho_saida = str(Path(__file__).resolve().parent / "video_tratado.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(caminho_saida, fourcc, fps, (960, 380), isColor=False)

    # pontos para correção de perspectiva
    pts_origem = np.float32([
        [120, 200],   # superior esquerdo
        [1160, 200],  # superior direito
        [1260, 600],  # inferior direito
        [20, 600]     # inferior esquerdo
    ])
    pts_destino = np.float32([
        [0, 0],
        [1200, 0],
        [1200, 480],
        [0, 480]
    ])
    matriz_perspectiva = cv2.getPerspectiveTransform(pts_origem, pts_destino)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # correção de perspectiva
        retificado = cv2.warpPerspective(
            frame, matriz_perspectiva, (1200, 480), flags=cv2.INTER_CUBIC
        )

        # recorte da região de interesse (ROI)
        roi = retificado[24:456, 36:1164]

        # padronização e interpolação da região de interesse
        padronizado = cv2.resize(roi, (960, 380), interpolation=cv2.INTER_AREA)

        # conversão para grayscale
        cinza = cv2.cvtColor(padronizado, cv2.COLOR_BGR2GRAY)

        # atenuação de ruído espacial (filtro gaussiano)
        filtrado = cv2.GaussianBlur(cinza, (3, 3), sigmaX=0.8)

        # salvar o frame tratado no novo arquivo de vídeo
        gravador.write(filtrado)

        # exibir os novos frames do vídeo tratados
        cv2.imshow("Video Original", frame)
        cv2.imshow("Video Tratado", filtrado)

        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    gravador.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
