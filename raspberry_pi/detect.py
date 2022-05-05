# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script principal para ejecutar la rutina de deteccion de objetos."""
import argparse
import sys
import time

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Ejecutar inferencias de forma continua en las imagenes adquiridas de la camara.

  Argumentos:
    model: Nombre del modelo de deteccion de objetos TFLite.
    camera_id: la identificacion de la camara que se pasara a OpenCV.
    width: El ancho del cuadro capturado desde la camara.
    height: La altura del cuadro capturado desde la camara.
    num_threads: el numero de subprocesos de la CPU para ejecutar el modelo.
    enable_edgetpu: Verdadero/Falso si el modelo es un modelo EdgeTPU.
  """

  # Variables para calcular FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Comience a capturar la entrada de video de la camara
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Parametros de visualizacion
  row_size = 20  # pixeles
  left_margin = 24  # pixeles
  text_color = (255, 0, 0)  # rojo
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Inicializar el modelo de deteccion de objetos
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # Capture continuamente imagenes de la camara y ejecute la inferencia
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: No se puede leer desde la camara web. Verifique la configuracion de su camara web.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    
    # Ejecute la estimacion de deteccion de objetos utilizando el modelo.
    detections = detector.detect(image)

    # Dibujar puntos clave y bordes en la imagen de entrada
    image = utils.visualize(image, detections)

    # Calcular la FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Mostrar la FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Detener el programa si se presiona la tecla ESC.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
