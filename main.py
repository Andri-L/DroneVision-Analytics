import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

input_video_path = '/content/DroneVision-Analytics/demo.mp4'          # Reemplaza con tu ruta de video de entrada
output_video_path = '/content/output_video.mp4'   # Reemplaza con la ruta deseada para la salida

class CountObject():
    def __init__(self, input_video_path, output_video_path) -> None:
        self.model = YOLO('yolov8s.pt')
        # Agregué 4 colores para 4 zonas; puedes personalizarlos
        self.colors = sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # Dimensiones del canvas (1920x1080, 16:9)
        width, height = 1920, 1080
        mid_x, mid_y = width // 2, height // 2

        # Definir 4 polígonos (cuadrantes) que juntos cubren exactamente el canvas
        self.polygons = [
            np.array([[0, 0], [mid_x, 0], [mid_x, mid_y], [0, mid_y]], np.int32),           # Zona superior izquierda
            np.array([[mid_x, 0], [width, 0], [width, mid_y], [mid_x, mid_y]], np.int32),       # Zona superior derecha
            np.array([[0, mid_y], [mid_x, mid_y], [mid_x, height], [0, height]], np.int32),       # Zona inferior izquierda
            np.array([[mid_x, mid_y], [width, mid_y], [width, height], [mid_x, height]], np.int32)  # Zona inferior derecha
        ]

        # Crear zonas a partir de los polígonos
        self.zones = [
            sv.PolygonZone(polygon=polygon)
            for polygon in self.polygons
        ]

        # Crear anotadores para cada zona
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=self.colors.by_idx(index),
                thickness=6,
                text_thickness=8,
                text_scale=4
            )
            for index, zone in enumerate(self.zones)
        ]

        # Crear anotadores para las detecciones (cajas) correspondientes a cada zona
        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors.by_idx(index),
                thickness=4,
            )
            for index in range(len(self.polygons))
        ]

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        # Realizar la detección de personas en el frame
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filtrar detecciones: clase 0 (personas) con confianza > 0.5
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

        # Recorrer cada zona para filtrar las detecciones y anotarlas
        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = zone_annotator.annotate(scene=frame)

        return frame

    def process_video(self):
        sv.process_video(source_path=self.input_video_path,
                         target_path=self.output_video_path,
                         callback=self.process_frame)

if __name__ == "__main__":
    obj = CountObject(input_video_path, output_video_path)
    obj.process_video()
