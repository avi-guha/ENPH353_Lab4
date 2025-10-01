#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np
import sys

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer for camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

        # SIFT detector
        self.sift = cv2.SIFT_create()

        self.template_img = None
        self.template_kp = None
        self.template_desc = None

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        # Load and process template
        self.template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        self.template_kp, self.template_desc = self.sift.detectAndCompute(self.template_img, None)

        self._is_template_loaded = True
        print("Loaded template image file: " + self.template_path)

    # Convert cv2 image to Qt Pixmap
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height,
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        if not ret:
            return

        display_frame = frame.copy()

        if self._is_template_loaded:
            # Convert to gray
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors in camera frame
            kp_frame, desc_frame = self.sift.detectAndCompute(gray_frame, None)

            if desc_frame is not None and len(desc_frame) > 0:
                # Match descriptors
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(self.template_desc, desc_frame, k=2)

                # Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                # Draw matches
                display_frame = cv2.drawMatches(self.template_img, self.template_kp,
                                                frame, kp_frame, good_matches, None,
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                # If enough matches, compute homography
                if len(good_matches) > 10:
                    src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h, w = self.template_img.shape
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        display_frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

        # Show result
        pixmap = self.convert_cv_to_pixmap(display_frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
