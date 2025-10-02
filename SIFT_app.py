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

    Behavior:
      - Main UI shows a clean live camera view.
      - Once a homography is successfully found (>= MIN_GOOD_MATCHES and H != None),
        the main UI draws ONLY the blue homography polygon on top of the live frame.
      - The popup always shows the side-by-side matches with lines; if homography is
        available, the polygon is drawn on the live (right) side in the popup too.

    Tuning:
      - Lowe's ratio is set to 0.60 (stricter).
      - Homography requires at least 40 "good" matches.
    """

    MIN_GOOD_MATCHES = 40
    RATIO_TEST = 0.60                 # stricter ratio per request
    RANSAC_REPROJ_THRESH = 5.0

    MATCH_W, MATCH_H = 900, 420       # popup canvas size (scaled to fit)
    CAM_W, CAM_H = 320, 240

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
        self._camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_W)
        self._camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)

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
        """Always-on-top window with match count and match-visualization canvas."""
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
        f = self.match_label.font()
        f.setPointSize(12)
        f.setBold(True)
        self.match_label.setFont(f)
        outer.addWidget(self.match_label)

        # Single canvas that will show cv2.drawMatches output
        self.matches_view = QtWidgets.QLabel()
        self.matches_view.setAlignment(QtCore.Qt.AlignCenter)
        self.matches_view.setFixedSize(self.MATCH_W, self.MATCH_H)
        self.matches_view.setStyleSheet("border:1px solid #888; background:#111; color:#ccc;")
        outer.addWidget(self.matches_view)

        self.match_dialog.hide()

    def _bgr_to_pixmap_scaled(self, bgr_img, max_w, max_h):
        """Convert BGR ndarray -> scaled QPixmap keeping aspect ratio."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)
        return pm.scaled(max_w, max_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def _update_match_popup(self, vis_bgr, good_count):
        """Update count text and visualization image in the popup."""
        self.match_label.setText(
            f"Good matches: {good_count} "
            f"({'OK' if good_count >= self.MIN_GOOD_MATCHES else 'need 40+'})"
        )
        if vis_bgr is not None:
            self.matches_view.setPixmap(self._bgr_to_pixmap_scaled(vis_bgr, self.MATCH_W, self.MATCH_H))

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

        # Show template in the main UI (no overlays)
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
            # Clear popup canvas
            blank = np.zeros(
                (self.CAM_H, self.CAM_W + (self.template_img_color.shape[1] if self.template_img_color is not None else self.CAM_W), 3),
                dtype=np.uint8
            )
            self._update_match_popup(blank, 0)

    def SLOT_query_camera(self):
        ok, frame = self._camera_device.read()
        if not ok:
            return

        # By default, main UI shows clean frame
        main_display = frame.copy()

        # --- POPUP: compute matches and draw only there ---
        vis_bgr = None
        good_matches = []
        homography_found = False
        proj_polygon = None  # to draw on main UI only if found

        if self._is_template_loaded:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, desc_frame = self.sift.detectAndCompute(gray, None)

            if desc_frame is not None and len(desc_frame) > 0:
                raw_matches = self.bf.knnMatch(self.template_desc, desc_frame, k=2)

                # Lowe ratio test @ 0.60
                for m, n in raw_matches:
                    if m.distance < self.RATIO_TEST * n.distance:
                        good_matches.append(m)

                # Prepare an overlay copy of the live frame for the POPUP only
                overlay_frame = frame.copy()

                # Try homography if enough matches
                if len(good_matches) >= self.MIN_GOOD_MATCHES:
                    src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.RANSAC_REPROJ_THRESH)
                    if H is not None:
                        h, w = self.template_img_gray.shape
                        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        proj = cv2.perspectiveTransform(corners, H)

                        # Draw polygon on the POPUP image (right side)
                        cv2.polylines(overlay_frame, [np.int32(proj)], True, (255, 0, 0), 3, cv2.LINE_AA)

                        # Save for MAIN UI draw (only polygon)
                        homography_found = True
                        proj_polygon = np.int32(proj)

                # Build the side-by-side matches WITH lines (and polygon if drawn above)
                vis_bgr = cv2.drawMatches(
                    self.template_img_color, self.template_kp,
                    overlay_frame, kp_frame if kp_frame is not None else [],
                    good_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.putText(
                    vis_bgr,
                    f"Good matches: {len(good_matches)} (need 40+ for Homography)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA
                )

        # Update popup (even if no descriptors found)
        if vis_bgr is None:
            blank = np.zeros(
                (self.CAM_H, self.CAM_W + (self.template_img_color.shape[1] if self.template_img_color is not None else self.CAM_W), 3),
                dtype=np.uint8
            )
            self._update_match_popup(blank, 0)
        else:
            self._update_match_popup(vis_bgr, len(good_matches))

        # --- MAIN UI: draw ONLY the polygon if homography was found; else keep clean ---
        if homography_found and proj_polygon is not None:
            cv2.polylines(main_display, [proj_polygon], True, (255, 0, 0), 3, cv2.LINE_AA)

        self.live_image_label.setPixmap(self.convert_cv_to_pixmap(main_display))

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
