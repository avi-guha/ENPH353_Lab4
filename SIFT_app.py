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
    - Popup shows current count of 'good matches' (Lowe ratio test).
    - When count >= MIN_GOOD_MATCHES (default 40), draw homography polygon
      around the detected template on the live frame; otherwise show
      side-by-side feature matches (like your first screenshot).

    References mirrored from your links:
      - SIFT:  https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
      - BF/ratio test: Pysource feature matching tutorial
      - Homography + RANSAC: Pysource object tracking with homography
    """

    MIN_GOOD_MATCHES = 40        # threshold you asked for
    RATIO_TEST = 0.75            # Lowe ratio
    RANSAC_REPROJ_THRESH = 5.0   # px

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

        # Timer to query camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

        # --- SIFT + Matcher ---
        # SIFT lives in opencv-contrib; this will raise if not installed.
        self.sift = cv2.SIFT_create(nfeatures=1500)
        # L2 for SIFT (float descriptors), no crossCheck when doing KNN
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Template fields
        self.template_path = None
        self.template_img_gray = None
        self.template_img_color = None
        self.template_kp = None
        self.template_desc = None

        # --- Non-modal popup that shows match count (always on top) ---
        self._build_match_popup()

    # ---------- UI helpers ----------

    def _build_match_popup(self):
        """Small tool window that we update each frame with the match count."""
        self.match_dialog = QtWidgets.QDialog(self)
        self.match_dialog.setWindowTitle("SIFT Matches")
        self.match_dialog.setWindowFlags(
            self.match_dialog.windowFlags()
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.match_dialog.setModal(False)
        lay = QtWidgets.QVBoxLayout(self.match_dialog)
        self.match_label = QtWidgets.QLabel("Good matches: 0")
        self.match_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.match_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.match_label.setFont(font)
        lay.addWidget(self.match_label)
        # Start hidden; we show it when the camera starts
        self.match_dialog.hide()

    def convert_cv_to_pixmap(self, cv_img_bgr):
        """cv2 BGR -> QPixmap"""
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

        # Show template on the left label (Qt side)
        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        # Load template (OpenCV side)
        self.template_img_gray = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        if self.template_img_gray is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to load template.")
            return

        # Keep a color copy for drawMatches side-by-side display
        self.template_img_color = cv2.cvtColor(self.template_img_gray, cv2.COLOR_GRAY2BGR)

        # Compute SIFT features for template
        self.template_kp, self.template_desc = self.sift.detectAndCompute(self.template_img_gray, None)
        self._is_template_loaded = self.template_desc is not None and len(self.template_desc) > 0

        print(f"Loaded template: {self.template_path}  (keypoints: {len(self.template_kp) if self.template_kp else 0})")

        if not self._is_template_loaded:
            QtWidgets.QMessageBox.information(self, "Template",
                                              "No SIFT features found in the template image.")
        else:
            QtWidgets.QMessageBox.information(self, "Template", "Template loaded and features computed.")

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
                # KNN match template->frame descriptors
                raw_matches = self.bf.knnMatch(self.template_desc, desc_frame, k=2)

                # Lowe ratio test
                for m, n in raw_matches:
                    if m.distance < self.RATIO_TEST * n.distance:
                        good_matches.append(m)

                # Update popup text
                self.match_label.setText(
                    f"Good matches: {len(good_matches)} "
                    f"({'OK' if len(good_matches) >= self.MIN_GOOD_MATCHES else 'need 40+'})"
                )

                if len(good_matches) >= self.MIN_GOOD_MATCHES:
                    # Compute homography (RANSAC) and draw polygon on live frame
                    src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.RANSAC_REPROJ_THRESH)
                    if H is not None:
                        h, w = self.template_img_gray.shape
                        template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        projected = cv2.perspectiveTransform(template_corners, H)

                        # Draw the homography polygon (blue) on the live frame
                        display_frame = cv2.polylines(display_frame, [np.int32(projected)],
                                                      isClosed=True, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

                        # Put a small text overlay too (handy while demoing)
                        cv2.putText(display_frame, f"Homography (good={len(good_matches)})",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        # Fall back to matches visualization if H failed
                        display_frame = self._draw_matches(self.template_img_color, self.template_kp,
                                                           frame, kp_frame, good_matches)
                else:
                    # Not enough matches: show side-by-side match lines
                    display_frame = self._draw_matches(self.template_img_color, self.template_kp,
                                                       frame, kp_frame, good_matches)

        # Push to the UI
        pixmap = self.convert_cv_to_pixmap(display_frame)
        self.live_image_label.setPixmap(pixmap)

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
            # Show the popup when the camera is active
            self.match_dialog.show()

    # ---------- Helpers ----------

    def _draw_matches(self, tmpl_bgr, tmpl_kp, frame_bgr, frame_kp, matches):
        """Side-by-side visualization of matches (like the first screenshot)."""
        vis = cv2.drawMatches(
            img1=tmpl_bgr, keypoints1=tmpl_kp,
            img2=frame_bgr, keypoints2=frame_kp if frame_kp is not None else [],
            matches1to2=matches, outImg=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        # add count overlay
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
