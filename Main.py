import sys
import threading
import time

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMessageBox
from deepface import DeepFace

from GUI import Ui_MainWindow


class SimpleThread(QThread):
    signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread_Active = None

    def run(self):
        self.thread_Active = True

        while self.thread_Active:
            time.sleep(0.1)
            self.signal.emit(1)

    def stop(self):
        self.thread_Active = False
        self.quit()


class Main:
    def __init__(self):

        self.dfs = []
        self.faces = []
        self.recognize_face = False
        self.camera_flag = None
        self.detection_frame = None
        self.stream_Image = None

        # Connecting the backend to the frontend
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        self.video_cap = cv2.VideoCapture(0)

        # Connect buttons to functions
        self.ui.pushButton.clicked.connect(self.add_person)
        self.ui.pushButton_2.clicked.connect(self.on_the_face_recognizer)
        self.ui.pushButton_3.clicked.connect(self.closeEvent)
        self.ui.pushButton_4.clicked.connect(self.login)

        # Background Thread to update the frames
        self.t = SimpleThread()
        self.t.signal.connect(self.update_thread_values)
        self.t.start()

        # Start the camera stream thread
        self.camera_thread = threading.Thread(target=self.start_camera_stream)
        self.camera_thread.start()

        self.detector_stream_thread = threading.Thread(target=self.detector_stream)
        self.detector_stream_thread.start()


    def login(self):
        username = self.ui.lineEdit_2.text()
        password = self.ui.lineEdit_3.text()

        if username == "" or password == "":
            QMessageBox.warning(self.MainWindow, "Warnings", "Credentials are missing")
            return

        if username == "admin" and password == "0000":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_3)
        else:
            QMessageBox.warning(self.MainWindow, "Warnings", "Wrong Credentials")
            return
    def closeEvent(self):
        self.camera_flag = False
        self.t.stop()
        self.MainWindow.close()

    def add_person(self):
        if self.ui.lineEdit.text() != "":
            name = self.ui.lineEdit.text()
            cv2.imwrite("KnownFace/" + name + ".jpg", self.stream_Image)
            QMessageBox.information(self.MainWindow, "Success", "Face added successfully!")
            self.ui.lineEdit.clear()
        else:
            QMessageBox.warning(self.MainWindow, "No Text Provided", "Please add a person name first")

    def detector_stream(self):

        while self.t.thread_Active:
            if type(self.stream_Image) == np.ndarray:
                self.detect_face()

            if self.recognize_face:
                self.recognize_person()



    def detect_face(self):
        self.faces = DeepFace.extract_faces(self.stream_Image, enforce_detection=False, align=True)

    def recognize_person(self):

        stream_Image = self.stream_Image.copy()
        self.dfs = DeepFace.find(stream_Image,
                                 db_path="KnownFace",
                                 model_name="ArcFace",  # For Face Recognition
                                 detector_backend="opencv",  # For Face Detection
                                 enforce_detection=False,
                                 distance_metric="cosine",
                                 silent=True)

    def on_the_face_recognizer(self):
        self.recognize_face = not self.recognize_face

        if self.recognize_face:
            self.ui.pushButton_2.setText("Stop Recognizer")
            self.ui.lineEdit.setVisible(False)
        else:
            self.ui.pushButton_2.setText("Start Recognizer")
            self.ui.lineEdit.setVisible(True)

    def stop_thread(self):
        self.camera_flag = False

    def start_camera_stream(self):

        vid = cv2.VideoCapture(0)
        while self.t.thread_Active:
            ret, self.stream_Image = vid.read()

    def update_thread_values(self, val):

        if type(self.stream_Image) == np.ndarray:

            if not self.recognize_face:
                for face in self.faces:
                    x, y, w, h = (face['facial_area']['x'],
                                  face['facial_area']['y'],
                                  face['facial_area']['w'],
                                  face['facial_area']['h'])
                    if x != 0 and y != 0:
                        cv2.rectangle(self.stream_Image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if self.recognize_face:
                for face in self.dfs:
                    if face.shape[0] > 0:
                        x, y, w, h = face.source_x[0], face.source_y[0], face.source_w[0], face.source_h[0]
                        if x != 0 and y != 0:
                            cv2.rectangle(self.stream_Image, (x, y), (x + w, y + h), (0, 255, 0), 4)
                            most_similar_face = face.iloc[0].identity
                            name = most_similar_face.split("\\")[-1][:-4]
                            cv2.putText(self.stream_Image, name, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)
                            cv2.putText(self.stream_Image, "Face Recognized Door Opened", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 0), 2)
                    else:
                        cv2.putText(self.stream_Image, "Unknown face Door Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 0), 2)

            rgb_frame = cv2.cvtColor(self.stream_Image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytesPerLine = ch * w
            convert_to_qt_format = QImage(rgb_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            img = convert_to_qt_format.scaled(w, h, Qt.KeepAspectRatio)
            self.ui.label_2.setPixmap(QPixmap.fromImage(img))

    def __del__(self):
        self.stop_thread()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Main()
    mainWindow.MainWindow.show()
    sys.exit(app.exec_())
