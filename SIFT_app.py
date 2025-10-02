#!/usr/bin/env python3
# Requires: pip install opencv-contrib-python PyQt5

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np
import sys


class My_App(QtWidgets.QMainWindow):
    """
    SIFT + BFMatcher + Homography (RANSAC).
    - Non-modal popup shows: (1) good match count, (2) template thumbnail, (3) live webcam thumbnail.
    - Homography polygon is drawn on the main live view when good matches >= 40.
    """

    MIN_GOOD_MATCHES = 40
    RATIO_TEST = 0.75
    RANSAC_REPROJ_THRESH = 5.0
    THUMB_W, THUMB_H = 220, 220  # popup thumbnail max size

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        # Camera config
        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self._camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

        # SIFT + matcher
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Template fields
        self.template_path = None
        self.template_img_gray = None
        self.template_img_color = None
        self.template_kp = None
        self.template_desc = None

        # Popup UI
        self._build_match_popup()

    # ---------- Popup construction & helpers ----------

    def _build_match_popup(self):
        """Small always-on-top tool window with match count + two thumbnails."""
        self.match_dialog = QtWidgets.QDialog(self)
        self.match_dialog.setWindowTitle("SIFT Matches")
        self.match_dialog.setWindowFlags(
            self.match_dialog.windowFlags()
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.match_dialog.setModal(False)

        outer = QtWidgets.QVBoxLayout(self.match_dialog)

        # Count label
        self.match_label = QtWidgets.QLabel("Good matches: 0 (need 40+)")
        self.match_label.setAlignment(QtCore.Qt.AlignCenter)
        f = self.match_label.font(); f.setPointSize(12); f.setBold(True)
        self.match_label.setFont(f)
        outer.addWidget(self.match_label)

        # Thumbnails row
        thumbs = QtWidgets.QHBoxLayout()
        self.tmpl_thumb = QtWidgets.QLabel("Template")
        self.live_thumb = QtWidgets.QLabel("Live")
        for lab in (self.tmpl_thumb, self.live_thumb):
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setFixedSize(self.THUMB_W, self.THUMB_H)
            lab.setStyleSheet("border:1px solid #888; background:#111; color:#ccc;")
        thumbs.addWidget(self.tmpl_thumb)
        thumbs.addWidget(self.live_thumb)
        outer.addLayout(thumbs)

        self.match_dialog.hide()

    def _bgr_to_pixmap_scaled(self, bgr_img, max_w, max_h):
        """Convert BGR ndarray -> scaled QPixmap keeping aspect ratio."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)
        return pm.scaled(max_w, max_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def _update_match_popup(self, frame_bgr, good_count):
        # Update count text
        self.match_label.setText(
            f"Good matches: {good_count} "
            f"({'OK' if good_count >= self.MIN_GOOD_MATCHES else 'need 40+'})"
        )

        # Template thumb
        if self.template_img_color is not None:
            self.tmpl_thumb.setPixmap(
                self._bgr_to_pixmap_scaled(self.template_img_color, self.THUMB_W, self.THUMB_H)
            )

        # Live thumb (current webcam frame)
        if frame_bgr is not None:
            self.live_thumb.setPixmap(
                self._bgr_to_pixmap_scaled(frame_bgr, self.THUMB_W, self.THUMB_H)
            )

    # ---------- UI helpers ----------

    def convert_cv_to_pixmap(self, cv_img_bgr):
        cv_img = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        q_img = QtGui.QImage(cv_img.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    # ---------- Slots ----------

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if not dlg.exec_():
            return
        self.template_path = dlg.selectedFiles()[0]

        # Show template in main UI
        self.template_label.setPixmap(QtGui.QPixmap(self.template_path))

        # Load template for OpenCV
        self.template_img_gray = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        if self.template_img_gray is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to load template.")
            return
        self.template_img_color = cv2.cvtColor(self.template_img_gray, cv2.COLOR_GRAY2BGR)

        # Precompute features
        self.template_kp, self.template_desc = self.sift.detectAndCompute(self.template_img_gray, None)
        self._is_template_loaded = self.template_desc is not None and len(self.template_desc) > 0

        print(f"Loaded template: {self.template_path} (keypoints: {len(self.template_kp) if self.template_kp else 0})")

        if not self._is_template_loaded:
            QtWidgets.QMessageBox.information(self, "Template", "No SIFT features found in the template image.")
        else:
            QtWidgets.QMessageBox.information(self, "Template", "Template loaded and features computed.")
            # Prime the popup template thumbnail
            self._update_match_popup(None, 0)

    def SLOT_query_camera(self):
        ok, frame = self._camera_device.read()
        if not ok:
            return

        display_frame = frame.copy()
        good_matches = []

        if self._is_template_loaded:
            # Detect features in live frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, desc_frame = self.sift.detectAndCompute(gray, None)

            if desc_frame is not None and len(desc_frame) > 0:
                # KNN match
                raw = self.bf.knnMatch(self.template_desc, desc_frame, k=2)

                # Lowe ratio test
                for m, n in raw:
                    if m.distance < self.RATIO_TEST * n.distance:
                        good_matches.append(m)

                # Update the popup thumbnails + count
                self._update_match_popup(frame, len(good_matches))

                # Draw on main view
                if len(good_matches) >= self.MIN_GOOD_MATCHES:
                    src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.RANSAC_REPROJ_THRESH)
                    if H is not None:
                        h, w = self.template_img_gray.shape
                        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        proj = cv2.perspectiveTransform(corners, H)
                        display_frame = cv2.polylines(display_frame, [np.int32(proj)],
                                                      True, (255, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(display_frame, f"Homography (good={len(good_matches)})",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        display_frame = self._draw_matches(self.template_img_color, self.template_kp,
                                                           frame, kp_frame, good_matches)
                else:
                    display_frame = self._draw_matches(self.template_img_color, self.template_kp,
                                                       frame, kp_frame, good_matches)
            else:
                # No descriptors found in the live frame
                self._update_match_popup(frame, 0)

        # Push to the main UI
        self.live_image_label.setPixmap(self.convert_cv_to_pixmap(display_frame))

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
            self.match_dialog.hide()
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")
            self.match_dialog.show()

    # ---------- Drawing helpers ----------

    def _draw_matches(self, tmpl_bgr, tmpl_kp, frame_bgr, frame_kp, matches):
        vis = cv2.drawMatches(
            tmpl_bgr, tmpl_kp,
            frame_bgr, frame_kp if frame_kp is not None else [],
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.putText(vis, f"Good matches: {len(matches)} (need 40+ for Homography)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
        return vis

    def closeEvent(self, event):
        try:
            if self._camera_device and self._camera_device.isOpened():
                self._camera_device.release()
        finally:
            event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
