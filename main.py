import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

input_video_path = '/content/DroneVision-Analytics/demo.mp4' # replace with the name of the video file
output_video_path = '/content/output_video.mp4'

class CountObject():
    def __init__(self, input_video_path, output_video_path) -> None:
        self.model = YOLO('yolov8s.pt')

        # adding colors for every polygon
        self.colors = sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # canva size
        width, height = 1920, 1080
        mid_x, mid_y = width // 2, height // 2

        # define 4 polygons (quadrants) that together cover the entire canvas
        self.polygons = [
            np.array([[0, 0], [mid_x, 0], [mid_x, mid_y], [0, mid_y]], np.int32),           # Zona superior izquierda
            np.array([[mid_x, 0], [width, 0], [width, mid_y], [mid_x, mid_y]], np.int32),       # Zona superior derecha
            np.array([[0, mid_y], [mid_x, mid_y], [mid_x, height], [0, height]], np.int32),       # Zona inferior izquierda
            np.array([[mid_x, mid_y], [width, mid_y], [width, height], [mid_x, height]], np.int32)  # Zona inferior derecha
        ]

        # make zones from the polygons defined above
        self.zones = [
            sv.PolygonZone(polygon=polygon)
            for polygon in self.polygons
        ]

        # make annotators for each zone
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

        # make annotators for each box in each zone
        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors.by_idx(index),
                thickness=4,
            )
            for index in range(len(self.polygons))
        ]

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        # perform person detection in the frame
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filter detections: class 0 (people) with confidence > 0.5
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

        # Iterate over each zone to filter the detections and annotate them
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
