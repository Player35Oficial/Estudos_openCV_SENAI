# -*- coding: utf-8 -*-
# Carregar as dependências
import cv2
import time

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

# Cores das classes
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# carrega AS CLASSES
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# captura do video
cap = cv2.VideoCapture(0)

# carregando os pesos da rede neural
net = cv2.dnn.readNet("/yolov3-tiny.weights", "/yolov3.cfg")

# Setando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)

# Lendo os frames do video 
while True:
    # Captura do Frame
    ret, frame = cap.read()

    if ret:  # if ret is True:
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # começo da contagem dos MS
    start = time.time()

    # detecção
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # fim da Contagem dos MS
    end = time.time()

    # Percorrer todas as detecções
    for (classid, score, box) in zip(classes, scores, boxes):
        # Gerando uma cor para a classe
        color = COLORS[int(classid) % len(COLORS)]

        # Pegando o nome da classe pelo ID e o seu score de acuracia
        label = f'({class_names[classid[0]]} : {score})'

        # Desenhando a box da detecção
        cv2.rectangle(frame, box, color, 2)

        # Escrevendo o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculando o tempo que levou para fazer a detecção
    fps_label = f"FPS: {round((1.0 / (end - start)), 2)}"

    # escrevendo o fps na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Mostrando a imagem
    cv2.imshow("detections", frame)

    # Espera da resposta
    if cv2.waitKey(1) == 27:
        break

# Liberação da câmera e destrói todas as janelas
cap.release()
cv2.destroyAllWindows()
