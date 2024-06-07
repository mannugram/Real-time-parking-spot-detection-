import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import threading
import psycopg2
from psycopg2.extras import execute_values

model = YOLO('yolov8s.pt')

class ParkingLotThread(threading.Thread):
    def __init__(self, video_path, lot_id):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.lot_id = lot_id
        self.empty_slots = 0
        self.running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Couldn't open video file for ParkingLotThread {self.lot_id}")
                return
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                time.sleep(1)
                frame = cv2.resize(frame, (1020, 500))

                results = model.predict(frame)
                px = pd.DataFrame(results[0].boxes.data).astype("float")
                list_slots = [0] * 12

                for index, row in px.iterrows():
                    x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
                    c = class_list[d]
                    if 'car' in c:
                        cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2
                        for i, area in enumerate(areas):
                            if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                                list_slots[i] += 1

                self.empty_slots = 12 - sum(list_slots)
                cv2.imshow(f"Parking Lot {self.lot_id}", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error in ParkingLotThread {self.lot_id}: {e}")
            self.running = False

    def stop(self):
        self.running = False

class DBWriterThread(threading.Thread):
    def __init__(self, parking_lots):
        threading.Thread.__init__(self)
        self.parking_lots = parking_lots
        self.running = True

    def run(self):
        conn = psycopg2.connect(dbname="mannraval", user="postgres", password="your_password", host="localhost", port="5432")
        while self.running:
            time.sleep(1)
            data = [(lot.lot_id, lot.empty_slots) for lot in self.parking_lots]
            with conn.cursor() as cur:
                query = """
                INSERT INTO parking_status (lot_id, empty_slots)
                VALUES %s
                ON CONFLICT (lot_id) 
                DO UPDATE SET empty_slots = EXCLUDED.empty_slots;
                """
                execute_values(cur, query, data)
                conn.commit()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    class_list = open("coco.txt").read().strip().split("\n")
    areas = [
        [(52, 364), (30, 417), (73, 412), (88, 369)],
        [(105, 353), (86, 428), (137, 427), (146, 358)],
        [(159, 354), (150, 427), (204, 425), (203, 353)],
        [(217, 352), (219, 422), (273, 418), (261, 347)],
        [(274, 345), (286, 417), (338, 415), (321, 345)],
        [(336, 343), (357, 410), (409, 408), (382, 340)],
        [(396, 338), (426, 404), (479, 399), (439, 334)],
        [(458, 333), (494, 397), (543, 390), (495, 330)],
        [(511, 327), (557, 388), (603, 383), (549, 324)],
        [(564, 323), (615, 381), (654, 372), (596, 315)],
        [(616, 316), (666, 369), (703, 363), (642, 312)],
        [(674, 311), (730, 360), (764, 355), (707, 308)],
    ]

    # Paths to your local video files
    video_files = [
        "parking1.mp4",
         
        # ... add paths to your other video files ...
    ]

    num_parking_lots = len(video_files)
    parking_lots = [ParkingLotThread(video_files[i], i + 1) for i in range(num_parking_lots)]

    for lot in parking_lots:
        lot.start()

    db_writer = DBWriterThread(parking_lots)
    db_writer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for lot in parking_lots:
            lot.stop()
        db_writer.stop()
        for lot in parking_lots:
            lot.join()
        db_writer.join()
        print("Program terminated gracefully.")
