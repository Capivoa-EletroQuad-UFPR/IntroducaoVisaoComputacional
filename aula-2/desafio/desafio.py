import sys
from pathlib import Path
import cv2
import numpy as np


def main():
    # carregar o vídeo no código e definir método inicial de limiarização
    caminho_video = str(Path(__file__).resolve().parent / "video.mp4")
    metodos = ["adaptativo", "otsu", "fixo"]
    metodo = "adaptativo"

    # leitura de argumentos opcionais (caminho do vídeo e/ou método)
    for arg in sys.argv[1:]:
        if arg.lower() in metodos:
            metodo = arg.lower()
        elif not arg.startswith("-"):
            caminho_video = arg

    if not Path(caminho_video).exists():
        candidatos = [
            Path(__file__).resolve().parent.parent.parent / "aula-1" / "desafio" / "video_tratado.mp4",
            Path(__file__).resolve().parent.parent.parent / "aula-1" / "desafio" / "video.mp4",
            Path("video.mp4")
        ]
        for c in candidatos:
            if c.exists():
                caminho_video = str(c)
                break

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {caminho_video}")
        return

    # carregar template para validação por template matching
    caminho_tpl = Path(__file__).resolve().parent / "template_objeto.png"
    template = cv2.imread(str(caminho_tpl), cv2.IMREAD_GRAYSCALE) if caminho_tpl.exists() else None

    # configuração para salvar o vídeo anotado
    caminho_saida = str(Path(__file__).resolve().parent / "video_anotado.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(caminho_saida, fourcc, fps, (960, 380))

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

    indice_metodo = metodos.index(metodo) if metodo in metodos else 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # retificação e padronização (caso o vídeo ainda não esteja retificado)
        h, w = frame.shape[:2]
        if (w, h) == (960, 380):
            padronizado = frame
        else:
            retificado = cv2.warpPerspective(
                frame, matriz_perspectiva, (1200, 480), flags=cv2.INTER_CUBIC
            )
            roi = retificado[24:456, 36:1164]
            padronizado = cv2.resize(roi, (960, 380), interpolation=cv2.INTER_AREA)

        # conversão para escala de cinza e atenuação de ruído espacial
        cinza = cv2.cvtColor(padronizado, cv2.COLOR_BGR2GRAY) if padronizado.ndim == 3 else padronizado
        suave = cv2.GaussianBlur(cinza, (7, 7), 0)

        # limiarização com o método selecionado
        metodo_atual = metodos[indice_metodo]
        if metodo_atual == "adaptativo":
            binaria = cv2.adaptiveThreshold(
                suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 12
            )
            desc_metodo = "Adaptativo Gaussiano"
        elif metodo_atual == "fixo":
            _, binaria = cv2.threshold(suave, 110, 255, cv2.THRESH_BINARY_INV)
            desc_metodo = "Limiar Fixo (T=110)"
        else:
            t_otsu, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            desc_metodo = f"Otsu (T={int(t_otsu)})"

        # fechamento morfológico para unir descontinuidades nas bordas dos objetos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

        # extração de contornos com hierarquia completa de parentesco (RETR_TREE)
        contornos, hierarquia = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        anotado = cv2.cvtColor(padronizado, cv2.COLOR_GRAY2BGR) if padronizado.ndim == 2 else padronizado.copy()

        bboxes_principais = []

        if hierarquia is not None and len(contornos) > 0:
            hier = hierarquia[0]

            # ordenar índices por área decrescente para priorizar objetos externos maiores
            indices_ordenados = sorted(
                range(len(contornos)),
                key=lambda idx: cv2.contourArea(contornos[idx]),
                reverse=True
            )

            for i in indices_ordenados:
                c = contornos[i]
                area = cv2.contourArea(c)
                if area < 1500:
                    continue

                parent = hier[i][3]
                x, y, w_box, h_box = cv2.boundingRect(c)

                # verifica se é furo interno pela hierarquia ou por contenção geométrica
                e_furo = (parent != -1)
                contido = any(
                    x >= bx and y >= by and (x + w_box) <= (bx + bw) and (y + h_box) <= (by + bh)
                    for bx, by, bw, bh in bboxes_principais
                )

                if e_furo or contido:
                    # desenha contorno do furo em laranja sem criar nova bounding box
                    cv2.drawContours(anotado, [c], -1, (0, 165, 255), 2)
                    continue

                bboxes_principais.append((x, y, w_box, h_box))

                # cálculo do centróide via momentos espaciais
                m = cv2.moments(c)
                cx = int(m["m10"] / m["m00"]) if m["m00"] != 0 else 0
                cy = int(m["m01"] / m["m00"]) if m["m00"] != 0 else 0

                # validação de classe com template matching
                valido = True
                if template is not None:
                    roi_obj = cinza[y:y + h_box, x:x + w_box]
                    th, tw = template.shape[:2]
                    if h_box >= 20 and w_box >= 20:
                        if h_box < th or w_box < tw:
                            escala = min(h_box / th, w_box / tw) * 0.95
                            tpl_alvo = cv2.resize(
                                template, (max(5, int(tw * escala)), max(5, int(th * escala)))
                            )
                        else:
                            tpl_alvo = template
                        res_tm = cv2.matchTemplate(roi_obj, tpl_alvo, cv2.TM_CCOEFF_NORMED)
                        valido = cv2.minMaxLoc(res_tm)[1] >= 0.50

                cor = (0, 230, 90) if valido else (0, 140, 255)

                # desenha contorno principal, caixa delimitadora e centróide
                cv2.drawContours(anotado, [c], -1, cor, 2)
                cv2.rectangle(anotado, (x, y), (x + w_box, y + h_box), cor, 2)
                cv2.circle(anotado, (cx, cy), 4, (0, 0, 255), -1)
                cv2.drawMarker(anotado, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

                # telemetria dimensional
                texto = f"({cx}, {cy}) | {w_box}x{h_box} | A:{int(area)}"
                cv2.putText(
                    anotado, texto, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA
                )

        # telemetria do método ativo
        cv2.putText(
            anotado, f"Metodo: {desc_metodo} [L: alternar]",
            (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA
        )

        # salvar frame anotado no novo vídeo
        gravador.write(anotado)

        # exibir os novos frames do vídeo tratados
        cv2.imshow("Mascara Binaria", binaria)
        cv2.imshow("Rastreamento e Inspecao", anotado)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), 27):
            break
        elif tecla in (ord("l"), ord("L")):
            indice_metodo = (indice_metodo + 1) % len(metodos)
            print(f"[Limiarização] Alternado para: {metodos[indice_metodo].upper()}")
        elif tecla == ord("1"):
            indice_metodo = 0
            print("[Limiarização] Alternado para: ADAPTATIVO")
        elif tecla == ord("2"):
            indice_metodo = 1
            print("[Limiarização] Alternado para: OTSU")
        elif tecla == ord("3"):
            indice_metodo = 2
            print("[Limiarização] Alternado para: FIXO")

    cap.release()
    gravador.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
