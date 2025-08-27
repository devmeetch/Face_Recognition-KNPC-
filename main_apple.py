import sys, os, json, cv2, face_recognition, numpy as np, datetime, hashlib, time, traceback, atexit
from threading import Lock
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QButtonGroup, QGroupBox, QRadioButton, QPushButton,
    QMessageBox, QFrame, QSizePolicy, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QFont, QPainter, QPen, QImage
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPoint, QSize, QFileSystemWatcher,
    qInstallMessageHandler, QtMsgType
)

# ------------------ CONFIG ------------------
DEBUG = True
DEBUG_LOG = "kiosk_debug.log"

# On macOS the built-in camera is almost always index 0
IS_MAC = sys.platform == "darwin"
CAM_INDEX = 1     # change if needed (USB cams may be 1/2)
TOLERANCE = 0.52
COUNTDOWN_SECS = 3
DUPLICATE_COOLDOWN_MS = 5000  # avoid instant re-trigger on same person

def dlog(msg):
    if not DEBUG:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def dcatch(where, ex):
    dlog(f"{where} ERROR: {ex}")
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
    except Exception:
        pass

def _qt_msg_handler(mode, context, message):
    tag = {
        QtMsgType.QtDebugMsg: "QT-DEBUG",
        getattr(QtMsgType, "QtInfoMsg", None): "QT-INFO",
        QtMsgType.QtWarningMsg: "QT-WARN",
        QtMsgType.QtCriticalMsg: "QT-CRIT",
        QtMsgType.QtFatalMsg: "QT-FATAL",
    }.get(mode, "QT")
    dlog(f"{tag}: {message}")

qInstallMessageHandler(_qt_msg_handler)

# ------------------ ASSETS ------------------
HERO_DEFAULT = "default_image.jpg"
LOGO_IMAGE   = "knpc_logo.png"
IMAGES_DIR   = "images"
DATASET_DIR  = "dataset"

# ------------------ SUBMISSION INDEX ------------------
ONCE_PER_DAY = True
INDEX_FILE = os.path.join("feedback_data", "submissions_index.json")

def _today_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def load_index():
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_index(idx):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def already_submitted(person_id):
    idx = load_index()
    if person_id not in idx:
        return False
    if ONCE_PER_DAY:
        return idx[person_id] == _today_str()
    return True

def mark_submitted(person_id):
    idx = load_index()
    idx[person_id] = _today_str()
    save_index(idx)

# ------------------ DLIB/face_recognition LOCK ------------------
DLIB_LOCK = Lock()

def fr_face_locations(img, **kwargs):
    with DLIB_LOCK:
        return face_recognition.face_locations(img, **kwargs)

def fr_face_encodings(img, known_face_locations=None, **kwargs):
    with DLIB_LOCK:
        return face_recognition.face_encodings(
            img, known_face_locations=known_face_locations, **kwargs
        )

# ------------------ CAMERA SINGLETON ------------------
class CameraSingleton:
    _cap = None
    _idx = None

    @classmethod
    def _open_backend_first(cls, idx):
        """
        Try platform-preferred backend first, then generic fallback.
        macOS  : AVFoundation
        others : try default (DShow used to be for Windows, but we avoid it here)
        """
        cap = None
        if IS_MAC:
            # AVFoundation is the correct backend on macOS
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            # Cross-platform safe default
            cap = cv2.VideoCapture(idx)
        return cap

    @classmethod
    def open(cls, index=CAM_INDEX):
        if cls._cap is not None and cls._cap.isOpened():
            return cls._cap

        for idx in (index, 1 - index, 2, 0, 1):  # try a few indices to be safe
            cap = cls._open_backend_first(idx)
            if cap and cap.isOpened():
                # Set some reasonable preferences; ignore if unsupported
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                except Exception:
                    pass
                try:
                    cap.set(cv2.CAP_PROP_FPS, 30)
                except Exception:
                    pass
                cls._cap, cls._idx = cap, idx
                dlog(f"FR: camera opened (persistent) at index {idx}")
                return cls._cap

        dlog("FR: camera open failed (persistent)")
        return None

    @classmethod
    def release(cls):
        try:
            if cls._cap is not None:
                cls._cap.release()
                dlog("FR: camera released at exit")
        except Exception:
            pass
        cls._cap = None

atexit.register(CameraSingleton.release)

# ------------------ SIGNATURE ------------------
class SignatureWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(340, 120)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self._blank = self.image.copy()
        self.last_point = QPoint()
        self.drawing = False

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) and self.drawing:
            if self.image.size() != self.size():
                self.image = QImage(self.size(), QImage.Format_RGB32)
                self.image.fill(Qt.white)
            p = QPainter(self.image)
            p.setPen(QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(self.last_point, e.pos())
            p.end()
            self.last_point = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, e):
        qp = QPainter(self)
        qp.drawImage(self.rect(), self.image, self.image.rect())
        qp.end()

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def save(self, filename):
        return self.image.save(filename)

    def is_empty(self):
        return self.image == self._blank

# ------------------ FACE RECO THREAD ------------------
class FaceRecognitionThread(QThread):
    match_found = pyqtSignal(str)  # dataset path
    no_match = pyqtSignal()

    def __init__(
        self,
        dataset_folder=DATASET_DIR,
        tolerance=TOLERANCE,
        max_seconds=10,
        camera_index=CAM_INDEX,
        stable_seconds=0.0,
        frame_scale=0.5,
        detection_model="hog",
        process_every_n=1,
        known_encodings=None,
        known_files=None,
    ):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.tolerance = float(tolerance)
        self.max_seconds = int(max_seconds)
        self.camera_index = int(camera_index)
        self.stable_seconds = float(stable_seconds)
        self.frame_scale = float(frame_scale)
        self.detection_model = detection_model
        self.process_every_n = int(process_every_n)
        self._running = True
        self._known_encodings = known_encodings
        self._known_files = known_files
        self.countdown_secs = COUNTDOWN_SECS
        try:
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            self.cascade = None

    def stop(self):
        self._running = False

    def _load_encodings(self):
        if (
            self._known_encodings is not None
            and self._known_files is not None
            and len(self._known_encodings) > 0
        ):
            dlog("FR: using provided encodings")
            return self._known_encodings, self._known_files

        encs, files = [], []
        if os.path.isdir(self.dataset_folder):
            for fn in sorted(os.listdir(self.dataset_folder)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    fp = os.path.join(self.dataset_folder, fn)
                    try:
                        img = face_recognition.load_image_file(fp)
                        boxes = fr_face_locations(img, number_of_times_to_upsample=1, model="hog")
                        enc = fr_face_encodings(img, known_face_locations=boxes or None, num_jitters=0)
                        if enc:
                            encs.append(enc[0]); files.append(fp)
                    except Exception as ex:
                        dcatch("FR._load_encodings", ex)
        dlog(f"FR: loaded {len(files)} dataset images")
        return encs, files

    def _stable_id(self, enc: np.ndarray) -> str:
        return "auto_" + hashlib.sha1(enc.tobytes()).hexdigest()[:16]

    def _save_to_dataset(self, rgb_full, person_id):
        os.makedirs(self.dataset_folder, exist_ok=True)
        out = os.path.join(self.dataset_folder, f"{person_id}.jpg")
        cv2.imwrite(out, cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR))
        dlog(f"FR: saved new face {out}")
        return out

    def _scale_locs_to_full(self, locs_small, scale):
        if scale == 1.0:
            return locs_small
        inv = 1.0 / scale
        return [(int(t * inv), int(r * inv), int(b * inv), int(l * inv)) for (t, r, b, l) in locs_small]

    def _countdown_enroll(self, cap):
        """3s headless countdown; capture only if exactly ONE face stays present; finalize with HOG upsample=1."""
        stable_start = None
        SCALE = 0.6
        t0 = time.time()
        while self._running and (time.time() - t0) < 12:
            ret, frame = cap.read()
            if not ret:
                continue

            face_ok = True
            if self.cascade is not None:
                small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, 1.12, 4, minSize=(80, 80))
                face_ok = (len(faces) == 1)

            if face_ok:
                if stable_start is None:
                    stable_start = time.time()
                    dlog("FR: countdown started (1 face)")
                if (time.time() - stable_start) >= self.countdown_secs:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locs = fr_face_locations(rgb, number_of_times_to_upsample=1, model="hog")
                    if len(locs) != 1:
                        dlog(f"FR: finalize check failed (faces={len(locs)})")
                        stable_start = None
                        continue
                    encs = fr_face_encodings(rgb, locs, num_jitters=1)
                    pid = self._stable_id(encs[0]) if encs else f"auto_{int(time.time())}"
                    return self._save_to_dataset(rgb, pid)
            else:
                if stable_start is not None:
                    dlog("FR: countdown reset (faces != 1)")
                stable_start = None

        dlog("FR: countdown timeout/no capture")
        return None

    def run(self):
        try:
            encs, files = self._load_encodings()
            cap = CameraSingleton.open(self.camera_index)
            if not cap:
                self.no_match.emit()
                return

            tickfreq = cv2.getTickFrequency()
            t0 = cv2.getTickCount() / tickfreq

            frame_idx = 0
            face_detected = False
            countdown_start = None
            frames_without_face = 0

            while self._running:
                now = cv2.getTickCount() / tickfreq
                if now - t0 > self.max_seconds:
                    dlog("FR: timed out")
                    break

                # Simple read is the most compatible across backends
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_idx += 1
                if self.process_every_n > 1 and (frame_idx % self.process_every_n) != 0:
                    continue

                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small = rgb_full if self.frame_scale == 1.0 else cv2.resize(
                    rgb_full, (0, 0), fx=self.frame_scale, fy=self.frame_scale
                )

                upsample = 0 if frames_without_face < 8 else 1
                face_locations = fr_face_locations(
                    small, number_of_times_to_upsample=upsample, model=self.detection_model
                )
                if not face_locations:
                    frames_without_face += 1
                    face_detected = False
                    countdown_start = None
                    continue
                frames_without_face = 0

                if not face_detected:
                    face_detected = True
                    countdown_start = now
                if self.stable_seconds > 0.0 and (now - countdown_start) < self.stable_seconds:
                    continue

                full_locs = self._scale_locs_to_full(face_locations, self.frame_scale)
                encodings = fr_face_encodings(rgb_full, full_locs or None, num_jitters=1)
                if not encodings:
                    face_detected = False
                    countdown_start = None
                    continue

                probe = encodings[0]

                if not encs:
                    dlog("FR: dataset empty -> countdown enroll")
                    p = self._countdown_enroll(cap)
                    if p: self.match_found.emit(p)
                    else: self.no_match.emit()
                    return

                dists = face_recognition.face_distance(encs, probe)
                best_idx = int(np.argmin(dists))
                best_dist = float(dists[best_idx])
                dlog(f"FR: best distance {best_dist:.3f}")

                if best_dist <= self.tolerance:
                    dlog("FR: match found")
                    self.match_found.emit(files[best_idx])
                    return
                else:
                    dlog("FR: no close match -> countdown enroll")
                    p = self._countdown_enroll(cap)
                    if p: self.match_found.emit(p)
                    else: self.no_match.emit()
                    return
        except Exception as ex:
            dcatch("FR.run", ex)
        finally:
            dlog("FR: thread exit")
            # keep camera open (persistent)

# ---- Background encoder ----
class EncodeWorker(QThread):
    done = pyqtSignal(object, object)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        encs, files = [], []
        try:
            if os.path.isdir(self.folder):
                for fn in sorted(os.listdir(self.folder)):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        fp = os.path.join(self.folder, fn)
                        try:
                            img = face_recognition.load_image_file(fp)
                            boxes = fr_face_locations(img, number_of_times_to_upsample=1, model="hog")
                            enc = fr_face_encodings(img, known_face_locations=boxes or None, num_jitters=0)
                            if enc:
                                encs.append(enc[0]); files.append(fp)
                        except Exception as ex:
                            dcatch("EncodeWorker.file", ex)
        except Exception as ex:
            dcatch("EncodeWorker.run", ex)
        self.done.emit(encs, files)

# ------------------ APP ------------------
class ReviewApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('KNPC Feedback Kiosk')
        self.setStyleSheet("background-color: #0f172a;")
        self.showFullScreen()

        self.matched_image_path = None
        self.matched_person_id = None

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(0)

        self.stack = QStackedWidget()
        self.root.addWidget(self.stack)

        self.welcome_page = self.build_welcome()
        self.stack.addWidget(self.welcome_page)

        self.survey_page = None

        self.slideshow_images = self.load_slideshow_images(IMAGES_DIR)
        self.slide_index = 0
        self.slideshow_timer = QTimer()
        self.slideshow_timer.timeout.connect(self.next_slide)
        QTimer.singleShot(100, self.start_slideshow)

        self.known_encodings, self.known_files = self.build_dataset_encodings(DATASET_DIR)

        self.recognition_thread = None
        self._is_starting = False
        self._restart_timer = None
        self._pending_dataset_refresh = False
        self._encoding_worker = None
        self._suppress_refresh_until = 0
        self._restart_after_encode = False

        self.start_recognition()

        self.fs_watcher = QFileSystemWatcher(self)
        if os.path.isdir(DATASET_DIR):
            self.fs_watcher.addPath(DATASET_DIR)
        self.fs_watcher.directoryChanged.connect(self._on_dataset_dir_changed)

        self._toast = QLabel(self)
        self._toast.setStyleSheet("""
            QLabel {
                background: rgba(15, 23, 42, 220);
                color: #e5e7eb;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 10px 16px;
                font: 16px 'Segoe UI';
            }
        """)
        self._toast.hide()

        dlog("UI: app started")

    # -------- util/UI --------
    def show_toast(self, text, ms=800):
        self._toast.setText(text)
        self._toast.adjustSize()
        x = (self.width() - self._toast.width()) // 2
        y = int(self.height() * 0.10)
        self._toast.move(max(10, x), max(10, y))
        self._toast.show()
        QTimer.singleShot(ms, self._toast.hide)

    def schedule_recognition_restart(self, delay_ms=300):
        dlog(f"UI: schedule_recognition_restart({delay_ms}ms)")
        self.stop_recognition()
        # If encoder is running, defer the restart until it's done
        if self._encoding_worker and self._encoding_worker.isRunning():
            self._restart_after_encode = True
            dlog("UI: restart deferred until EncodeWorker finishes")
            return
        if self._restart_timer is not None:
            try:
                self._restart_timer.stop()
                self._restart_timer.deleteLater()
            except Exception:
                pass
            self._restart_timer = None
        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.timeout.connect(self._start_recognition_guarded)
        self._restart_timer.start(max(0, int(delay_ms)))

    def _start_recognition_guarded(self):
        try:
            self.start_recognition()
        finally:
            if self._restart_timer is not None:
                try:
                    self._restart_timer.deleteLater()
                except Exception:
                    pass
                self._restart_timer = None

    # -------- encodings --------
    def build_dataset_encodings(self, folder):
        encs, files = [], []
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    fp = os.path.join(folder, fn)
                    try:
                        img = face_recognition.load_image_file(fp)
                        boxes = fr_face_locations(img, number_of_times_to_upsample=1, model="hog")
                        enc = fr_face_encodings(img, known_face_locations=boxes or None, num_jitters=0)
                        if enc:
                            encs.append(enc[0]); files.append(fp)
                    except Exception as ex:
                        dcatch("build_dataset_encodings", ex)
        dlog(f"UI: initial encodings {len(files)}")
        return encs, files

    def refresh_dataset_encodings(self, force=False):
        if not force and self.stack.currentWidget() != self.welcome_page:
            self._pending_dataset_refresh = True
            return
        if self._encoding_worker and self._encoding_worker.isRunning():
            self._pending_dataset_refresh = True
            return
        self._encoding_worker = EncodeWorker(DATASET_DIR)
        self._encoding_worker.done.connect(self._on_encodings_ready, Qt.QueuedConnection)
        self._encoding_worker.start()

    def _on_encodings_ready(self, encs, files):
        self.known_encodings, self.known_files = encs, files
        dlog(f"UI: encodings refreshed -> {len(files)} files")
        if self.stack.currentWidget() == self.welcome_page:
            if self._restart_after_encode:
                self._restart_after_encode = False
                self.start_recognition()
            else:
                self.schedule_recognition_restart(200)
            self.show_toast("Dataset updated", ms=600)
        if self._pending_dataset_refresh:
            self._pending_dataset_refresh = False
            self.refresh_dataset_encodings(force=True)

    # -------- layout --------
    def build_welcome(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(60, 40, 60, 24)
        lay.setSpacing(10)

        header = QWidget()
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        h.setAlignment(Qt.AlignCenter)

        logo = QLabel()
        pm = QPixmap(LOGO_IMAGE)
        if not pm.isNull():
            logo.setPixmap(pm.scaled(QSize(56, 56), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        h.addWidget(logo)

        title = QLabel("Your Feedback Matters")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #e5e7eb;")
        title.setFont(QFont('Segoe UI', 40, QFont.Black))
        h.addWidget(title)

        lay.addWidget(header)

        subtitle = QLabel("We appreciate your valuable feedback.\nPlease look at the camera to get started.")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#cbd5e1;")
        subtitle.setFont(QFont('Segoe UI', 18))
        lay.addWidget(subtitle)

        self.slide_label = QLabel()
        self.slide_label.setAlignment(Qt.AlignCenter)
        self.slide_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.slide_label.setMinimumSize(600, 360)
        self.slide_label.setStyleSheet("""
            QLabel {
                border-radius: 16px;
                border: 1px solid #3b4153;
                background: #0b1220;
            }
        """)
        lay.addWidget(self.slide_label, stretch=1, alignment=Qt.AlignCenter)

        footer_logo = QLabel()
        if not pm.isNull():
            footer_logo.setPixmap(pm.scaled(QSize(160, 80), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        footer_logo.setAlignment(Qt.AlignCenter)
        footer_logo.setStyleSheet("margin-top: 18px; margin-bottom: 8px;")
        lay.addWidget(footer_logo)

        return w

    # -------- slideshow --------
    def load_slideshow_images(self, folder):
        if not os.path.exists(folder):
            return [HERO_DEFAULT]
        pics = [os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        return pics or [HERO_DEFAULT]

    def start_slideshow(self):
        dlog("UI: slideshow start")
        self.update_slide()
        self.slideshow_timer.start(3000)

    def update_slide(self):
        if not hasattr(self, "slide_label"):
            return
        sz = self.slide_label.size()
        if sz.width() < 2 or sz.height() < 2:
            return
        img_path = self.slideshow_images[self.slide_index]
        pix = QPixmap(img_path)
        if pix.isNull():
            pix = QPixmap(HERO_DEFAULT)
        self.slide_label.setPixmap(
            pix.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def next_slide(self):
        self.slide_index = (self.slide_index + 1) % len(self.slideshow_images)
        self.update_slide()

    # -------- recognition control --------
    def start_recognition(self, *, fast=True):
        if self._is_starting:
            return
        self._is_starting = True
        try:
            if self.recognition_thread and self.recognition_thread.isRunning():
                self.stop_recognition()

            if not CameraSingleton.open(CAM_INDEX):
                dlog("UI: camera unavailable")
                return

            params = dict(
                dataset_folder=DATASET_DIR,
                tolerance=TOLERANCE,
                max_seconds=10,
                camera_index=CAM_INDEX,
                stable_seconds=0.0,
                frame_scale=0.5,
                detection_model="hog",
                process_every_n=1,
                known_encodings=self.known_encodings,
                known_files=self.known_files,
            )
            self.recognition_thread = FaceRecognitionThread(**params)
            self.recognition_thread.match_found.connect(self.on_face_matched, Qt.QueuedConnection)
            self.recognition_thread.no_match.connect(self.on_no_match, Qt.QueuedConnection)
            self.recognition_thread.finished.connect(self._on_recognition_finished, Qt.QueuedConnection)
            self.recognition_thread.start()
            dlog("UI: recognition started")
        except Exception as ex:
            dcatch("start_recognition", ex)
        finally:
            self._is_starting = False

    def _on_recognition_finished(self):
        self.recognition_thread = None
        dlog("UI: recognition finished")

    def stop_recognition(self):
        if self.recognition_thread and self.recognition_thread.isRunning():
            try:
                self.recognition_thread.stop()
                self.recognition_thread.wait(1500)
            except Exception as ex:
                dcatch("stop_recognition", ex)
        self.recognition_thread = None

    def _on_dataset_dir_changed(self, _path):
        dlog("UI: dataset directoryChanged -> refresh")
        if time.time() < self._suppress_refresh_until:
            dlog("UI: refresh suppressed briefly")
            return
        QTimer.singleShot(0, self.refresh_dataset_encodings)

    # -------- events --------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_slide()

    def closeEvent(self, e):
        dlog("UI: closeEvent ignored (kiosk)")
        e.ignore()

    # -------- matched / duplicate handling --------
    def _append_encoding_if_new(self, path):
        try:
            if path in self.known_files:
                return
            img = face_recognition.load_image_file(path)
            boxes = fr_face_locations(img, number_of_times_to_upsample=1, model="hog")
            enc = fr_face_encodings(img, known_face_locations=boxes or None, num_jitters=0)
            if enc:
                self.known_encodings.append(enc[0])
                self.known_files.append(path)
                dlog(f"UI: appended encoding for {os.path.basename(path)} (now {len(self.known_files)})")
        except Exception as ex:
            dcatch("_append_encoding_if_new", ex)

    def on_face_matched(self, matched_path):
        try:
            self._suppress_refresh_until = time.time() + 2
            person_id = os.path.splitext(os.path.basename(matched_path))[0]
            dlog(f"UI: on_face_matched -> {person_id}")

            self._append_encoding_if_new(matched_path)

            if already_submitted(person_id):
                dlog("UI: duplicate submit -> cooldown and welcome")
                self.show_toast("Thank you for submitting your review today.", ms=1500)
                self.stack.setCurrentWidget(self.welcome_page)
                self.slide_index = 0
                self.update_slide()
                if not self.slideshow_timer.isActive():
                    self.slideshow_timer.start(3000)
                self.schedule_recognition_restart(DUPLICATE_COOLDOWN_MS)
                return

            self.matched_image_path = matched_path
            self.matched_person_id = person_id
            self.show_survey_screen()
            self.refresh_dataset_encodings(force=True)
        except Exception as e:
            dcatch("on_face_matched", e)

    def on_no_match(self):
        if self.stack.currentWidget() == self.welcome_page:
            dlog("UI: no_match -> restart")
            self.schedule_recognition_restart(200)

    # -------- pages --------
    def show_survey_screen(self):
        dlog("UI: show_survey_screen")
        self.slideshow_timer.stop()
        self.stop_recognition()

        if self.survey_page is not None:
            idx = self.stack.indexOf(self.survey_page)
            if idx != -1:
                self.stack.removeWidget(self.survey_page)
            self.survey_page.deleteLater()
            self.survey_page = None

        self.survey_page = QWidget()
        main_layout = QHBoxLayout(self.survey_page)
        main_layout.setContentsMargins(60, 60, 60, 30)
        main_layout.setSpacing(40)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        lbl_match = QLabel("<b>Matched Image:</b>")
        lbl_match.setStyleSheet("color:#e5e7eb;")
        left_layout.addWidget(lbl_match, alignment=Qt.AlignLeft)

        matched_img_label = QLabel()
        matched_img_label.setFixedSize(330, 390)
        matched_img_label.setAlignment(Qt.AlignCenter)
        matched_img_label.setStyleSheet("""
            border-radius: 16px;
            border: 1px solid #334155;
            background: #0b1220;
        """)
        pixmap = QPixmap(self.matched_image_path or HERO_DEFAULT)
        matched_img_label.setPixmap(pixmap.scaled(matched_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        left_layout.addWidget(matched_img_label, alignment=Qt.AlignLeft)

        lbl_sign = QLabel("<b>Signature:</b>")
        lbl_sign.setStyleSheet("color:#e5e7eb; margin-top: 6px;")
        left_layout.addWidget(lbl_sign, alignment=Qt.AlignLeft)

        self.signature_widget = SignatureWidget(self)
        left_layout.addWidget(self.signature_widget, alignment=Qt.AlignLeft)

        clear_btn = QPushButton("Clear Signature")
        clear_btn.setStyleSheet("""
            QPushButton { background:#e2e8f0; color:#0f172a; border-radius:8px; padding:6px 14px; }
            QPushButton:hover { background:#cbd5e1; }
        """)
        clear_btn.clicked.connect(self.signature_widget.clear)
        left_layout.addWidget(clear_btn, alignment=Qt.AlignLeft)
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.setSpacing(14)
        self.button_groups = []

        questions = [
            "How was the exhibition?",
            "Rate the display quality:",
            "Would you recommend it?",
        ]
        options = [
            ["Excellent", "Good", "Average", "Poor"],
            ["Excellent", "Good", "Average", "Poor"],
            ["Yes", "No"],
        ]
        for q, opts in zip(questions, options):
            groupbox = QGroupBox(q)
            groupbox.setFont(QFont('Segoe UI', 18, QFont.DemiBold))
            groupbox.setStyleSheet("""
                QGroupBox {
                    color:#e5e7eb;
                    border-radius: 12px;
                    background: #111827;
                    padding: 16px;
                    border: 1px solid #374151;
                }
            """)
            btn_layout = QHBoxLayout()
            btn_group = QButtonGroup(self)
            for opt in opts:
                btn = QRadioButton(opt)
                btn.setFont(QFont('Segoe UI', 16))
                btn.setStyleSheet("""
                    QRadioButton {
                        padding: 12px 22px;
                        border-radius: 12px;
                        background: #0b1220;
                        color: #e5e7eb;
                        margin: 8px 12px;
                        min-width: 120px;
                        min-height: 46px;
                        border: 1px solid #334155;
                    }
                    QRadioButton::indicator { width: 0px; height: 0px; }
                    QRadioButton:checked {
                        background: #0ea5a5;
                        color: white;
                        border: 1px solid #0ea5a5;
                    }
                """)
                btn_group.addButton(btn)
                btn_layout.addWidget(btn)
            groupbox.setLayout(btn_layout)
            right_layout.addWidget(groupbox)
            self.button_groups.append(btn_group)

        right_layout.addStretch()

        submit_btn = QPushButton("Submit Review")
        submit_btn.setFont(QFont('Segoe UI', 18, QFont.Bold))
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #007A33;
                color: white;
                border-radius: 12px;
                padding: 14px 36px;
                font-size: 18px;
            }
            QPushButton:hover { background-color: #059b44; }
        """)
        submit_btn.clicked.connect(self.submit_review)
        right_layout.addWidget(submit_btn, alignment=Qt.AlignRight)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        footer_line = QFrame()
        footer_line.setFrameShape(QFrame.HLine)
        footer_line.setStyleSheet("color: #334155; background:#334155; margin: 0 60px 10px 60px;")

        footer_logo = QLabel()
        pm = QPixmap(LOGO_IMAGE)
        if not pm.isNull():
            footer_logo.setPixmap(pm.scaled(QSize(180, 90), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        footer_logo.setAlignment(Qt.AlignCenter)
        footer_logo.setStyleSheet("margin-bottom: 8px;")

        outer = QVBoxLayout()
        outer.addWidget(self.survey_page)
        outer.addWidget(footer_line)
        outer.addWidget(footer_logo)

        container = QWidget()
        container.setLayout(outer)
        self.stack.addWidget(container)
        self.stack.setCurrentWidget(container)

        self.survey_container = container
        dlog("UI: survey screen shown")

    def show_welcome_screen(self):
        dlog("UI: show_welcome_screen")
        self._suppress_refresh_until = time.time() + 1
        self.stack.setCurrentWidget(self.welcome_page)

        self.slide_index = 0
        self.update_slide()
        if not self.slideshow_timer.isActive():
            self.slideshow_timer.start(3000)

        self.schedule_recognition_restart(200)

        if hasattr(self, "survey_container") and self.survey_container is not None:
            idx = self.stack.indexOf(self.survey_container)
            if idx != -1:
                self.stack.removeWidget(self.survey_container)
            if self.survey_page is not None:
                self.survey_page.deleteLater()
                self.survey_page = None
            self.survey_container.deleteLater()
            self.survey_container = None

        if self._pending_dataset_refresh:
            self._pending_dataset_refresh = False
            self.refresh_dataset_encodings(force=True)

    # -------- save --------
    def submit_review(self):
        dlog("UI: submit_review clicked")
        answers = []
        for group in self.button_groups:
            btn = group.checkedButton()
            if btn is None:
                QMessageBox.warning(self, "Incomplete", "Please select all answers before submitting.")
                return
            answers.append(btn.text())

        if self.signature_widget.is_empty():
            QMessageBox.warning(self, "Signature Missing", "Please provide your signature before submitting.")
            return

        person_id = self.matched_person_id or ("unknown_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        nowstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("feedback_data", nowstamp)
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "review.txt"), "w", encoding="utf-8") as f:
            for q, a in zip(
                ["How was the exhibition?", "Rate the display quality:", "Would you recommend it?"],
                answers,
            ):
                f.write(f"{q}: {a}\n")
            f.write(f"person_id: {person_id}\n")
            f.write(f"timestamp: {nowstamp}\n")

        if self.matched_image_path and os.path.exists(self.matched_image_path):
            QPixmap(self.matched_image_path).save(os.path.join(save_dir, "matched_image.jpg"))
        self.signature_widget.save(os.path.join(save_dir, "signature.png"))

        mark_submitted(person_id)

        self.show_toast("Review submitted", ms=600)
        self.show_welcome_screen()

# ------------------ MAIN ------------------
if __name__ == '__main__':
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)

        def _excepthook(exctype, value, tb):
            dcatch("UNCAUGHT", value)
        sys.excepthook = _excepthook

        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        window = ReviewApp()
        window.show()
        rc = app.exec_()
        dlog(f"MAIN: app.exec_() returned {rc}")
        sys.exit(rc)
    except Exception as ex:
        dcatch("MAIN", ex)
        raise
