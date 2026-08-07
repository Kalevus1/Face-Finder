# -*- coding: utf-8 -*-


import os
import sys
import shutil
from datetime import datetime

import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QRect, QSize, QPoint
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QRadioButton,
    QCheckBox, QButtonGroup, QProgressBar, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QScrollArea, QFrame, QLayout, QSizePolicy,
    QDialog, QToolButton,
)

import malla_facial as malla

# ------------------- Rutas y configuración -------------------
# Al ejecutarse como .exe (PyInstaller) los datos van dentro del paquete (sys._MEIPASS);
# como script normal, junto a este archivo.
if getattr(sys, "frozen", False):
    BASE = sys._MEIPASS
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
MODELO_DET = os.path.join(BASE, "modelos", "face_detection_yunet_2023mar.onnx")
MODELO_REC = os.path.join(BASE, "modelos", "face_recognition_sface_2021dec.onnx")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
        ".heic", ".heif", ".avif"}                       # incluye formatos de Apple
UMBRALES = {"Estricta": 0.40, "Normal": 0.363, "Amplia": 0.30}
LADO_MAX = 1600
MINI = 132

# --------- carga de imagen con orientación EXIF (Pillow) ---------
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()        # habilita HEIC / HEIF / AVIF (fotos de Apple)
    SOPORTA_APPLE = True
except Exception:
    SOPORTA_APPLE = False


def cargar_imagen(ruta):
    try:
        with Image.open(ruta) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.array(im)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def listar_imagenes(carpeta):
    rutas = []
    for raiz, _, archivos in os.walk(carpeta):
        for n in archivos:
            if os.path.splitext(n)[1].lower() in EXTS:
                rutas.append(os.path.join(raiz, n))
    return rutas


def recortar_rostro(img, cara, margen=0.6):
    """Recorta la región de un rostro con margen. Devuelve (recorte, (x0, y0))."""
    h, w = img.shape[:2]
    x, y, bw, bh = cara[:4]
    mx, my = bw * margen, bh * margen
    x0 = max(0, int(x - mx)); y0 = max(0, int(y - my))
    x1 = min(w, int(x + bw + mx)); y1 = min(h, int(y + bh + my))
    return img[y0:y1, x0:x1].copy(), (x0, y0)


def resize_alto(img, alto):
    h, w = img.shape[:2]
    s = alto / max(1, h)
    return cv2.resize(img, (max(1, int(w * s)), alto))


def dibujar_5puntos_crop(crop, cara, offset):
    """Respaldo si MediaPipe falla: dibuja los 5 puntos de YuNet sobre el recorte."""
    out = crop.copy()
    x0, y0 = offset
    for px, py in cara[4:14].reshape(5, 2):
        cv2.circle(out, (int(px - x0), int(py - y0)), 3, (238, 211, 34), -1, cv2.LINE_AA)
    return out


def bgr_a_pixmap(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def miniatura_pixmap(ruta, lado=MINI):
    img = cargar_imagen(ruta)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = lado / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    return bgr_a_pixmap(img)


# ------------------- Motor de reconocimiento -------------------
class MotorFacial:
    def __init__(self):
        self.det = cv2.FaceDetectorYN_create(
            MODELO_DET, "", (320, 320),
            score_threshold=0.8, nms_threshold=0.3, top_k=5000)
        self.rec = cv2.FaceRecognizerSF_create(MODELO_REC, "")

    def _redim(self, img):
        h, w = img.shape[:2]
        lado = max(h, w)
        if lado <= LADO_MAX:
            return img
        e = LADO_MAX / lado
        return cv2.resize(img, (int(w * e), int(h * e)))

    def _detectar(self, img):
        img = self._redim(img)
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, caras = self.det.detect(img)
        return img, caras

    def rostros(self, img):
        img, caras = self._detectar(img)
        out = []
        if caras is None:
            return out
        for cara in caras:
            try:
                out.append(self.rec.feature(self.rec.alignCrop(img, cara)))
            except Exception:
                continue
        return out

    def principal(self, img):
        img, caras = self._detectar(img)
        if caras is None or len(caras) == 0:
            return None, None, img
        cara = max(caras, key=lambda c: c[2] * c[3])
        return self.rec.feature(self.rec.alignCrop(img, cara)), cara, img

    def parecido(self, a, b):
        return self.rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE)


class Motor:
    """Contenedor del motor (se crea una vez, en segundo plano)."""

    def __init__(self):
        self.facial = MotorFacial()

    def detectar(self, ruta):
        """Detecta TODOS los rostros. Devuelve (img_det, caras, recortes)."""
        img = cargar_imagen(ruta)
        if img is None:
            raise ValueError("No pude leer esa imagen.")
        img_det, caras = self.facial._detectar(img)
        if caras is None or len(caras) == 0:
            raise ValueError("No detecté ninguna cara.\nUsa una foto clara y de frente.")
        # más grande primero (suele ser la persona principal)
        caras = np.array(sorted(caras, key=lambda c: c[2] * c[3], reverse=True))
        recortes = [recortar_rostro(img_det, c, 0.25)[0] for c in caras]
        return img_det, caras, recortes

    def analizar(self, img_det, caras, idx):
        """Calcula la huella del rostro elegido y dibuja su malla neón."""
        cara = caras[idx]
        huella = self.facial.rec.feature(self.facial.rec.alignCrop(img_det, cara))
        crop, offset = recortar_rostro(img_det, cara, 0.6)
        mesh = None
        try:                                         # la malla es solo estética
            det = malla.crear_detector_malla()       # se crea en este hilo
            try:
                mesh = malla.dibujar_malla_jarvis(crop, det)
            finally:
                det.close()
        except Exception:
            mesh = None
        if mesh is None:                             # respaldo: 5 puntos de YuNet
            mesh = dibujar_5puntos_crop(crop, cara, offset)
        return huella, mesh


def copiar_seguro(ruta, destino):
    base = os.path.basename(ruta)
    nombre, ext = os.path.splitext(base)
    final = os.path.join(destino, base)
    c = 1
    while os.path.exists(final):
        final = os.path.join(destino, f"{nombre}_{c}{ext}")
        c += 1
    try:
        shutil.copy2(ruta, final)
    except Exception:
        pass


# ------------------- Hilos (QThread) -------------------
class HiloInit(QThread):
    listo = Signal(object)

    def run(self):
        self.listo.emit(Motor())


class HiloDetectar(QThread):
    listo = Signal(object, object, object)   # img_det, caras, recortes
    error = Signal(str)

    def __init__(self, motor, ruta):
        super().__init__()
        self.motor, self.ruta = motor, ruta

    def run(self):
        try:
            img_det, caras, recortes = self.motor.detectar(self.ruta)
            self.listo.emit(img_det, caras, recortes)
        except Exception as e:
            self.error.emit(str(e))


class HiloAnalizar(QThread):
    listo = Signal(object, object)     # huella, mesh(bgr)
    error = Signal(str)

    def __init__(self, motor, img_det, caras, idx):
        super().__init__()
        self.motor, self.img_det, self.caras, self.idx = motor, img_det, caras, idx

    def run(self):
        try:
            huella, mesh = self.motor.analizar(self.img_det, self.caras, self.idx)
            self.listo.emit(huella, mesh)
        except Exception as e:
            self.error.emit(str(e))


class DialogoRostros(QDialog):
    """Muestra los rostros detectados para que el usuario elija cuál analizar."""

    def __init__(self, recortes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Elegir rostro")
        self.idx = -1
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)
        info = QLabel(f"Se detectaron {len(recortes)} rostros.\nHaz clic en el que quieres buscar:")
        info.setObjectName("h1")
        info.setWordWrap(True)
        lay.addWidget(info)

        cont = QWidget()
        fila = QHBoxLayout(cont)
        fila.setSpacing(12)
        for i, crop in enumerate(recortes):
            pix = bgr_a_pixmap(resize_alto(crop, 160))
            tb = QToolButton()
            tb.setObjectName("facechoice")
            tb.setIcon(QIcon(pix))
            tb.setIconSize(pix.size())
            tb.setText(f"Rostro {i + 1}")
            tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            tb.setCursor(Qt.CursorShape.PointingHandCursor)
            tb.clicked.connect(lambda _=False, k=i: self._elegir(k))
            fila.addWidget(tb)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cont)
        scroll.setFixedHeight(240)
        scroll.setMinimumWidth(min(900, 200 * len(recortes) + 40))
        lay.addWidget(scroll)

    def _elegir(self, i):
        self.idx = i
        self.accept()


class HiloBusqueda(QThread):
    total = Signal(int)
    progreso = Signal(int, str)
    encontrada = Signal(str)
    terminado = Signal(str)            # "ok" / "cancelado"
    error = Signal(str)

    def __init__(self, motor, huella, carpeta, umbral, copiar):
        super().__init__()
        self.motor, self.huella = motor, huella
        self.carpeta, self.umbral, self.copiar = carpeta, umbral, copiar
        self._cancel = False

    def cancelar(self):
        self._cancel = True

    def run(self):
        try:
            imagenes = listar_imagenes(self.carpeta)
            total = len(imagenes)
            if total == 0:
                self.error.emit("No encontré imágenes en esa carpeta.")
                return
            destino = None
            if self.copiar:
                marca = datetime.now().strftime("%Y%m%d_%H%M%S")
                destino = os.path.join(self.carpeta, f"_Encontradas_{marca}")
                os.makedirs(destino, exist_ok=True)
            self.destino = destino
            self.total.emit(total)
            for i, ruta in enumerate(imagenes, start=1):
                if self._cancel:
                    self.terminado.emit("cancelado")
                    return
                img = cargar_imagen(ruta)
                if img is not None:
                    for huella in self.motor.facial.rostros(img):
                        if self._cancel:
                            break
                        if self.motor.facial.parecido(self.huella, huella) >= self.umbral:
                            if destino:
                                copiar_seguro(ruta, destino)
                            self.encontrada.emit(ruta)
                            break
                self.progreso.emit(i, os.path.basename(ruta))
            self.terminado.emit("ok")
        except Exception as e:
            self.error.emit(str(e))


# ------------------- FlowLayout (envuelve por filas) -------------------
class FlowLayout(QLayout):
    def __init__(self, parent=None, margen=10, espacio=12):
        super().__init__(parent)
        self.setContentsMargins(margen, margen, margen, margen)
        self._esp = espacio
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, ancho):
        return self._acomodar(QRect(0, 0, ancho, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._acomodar(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _acomodar(self, rect, solo_prueba):
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        alto_fila = 0
        derecha = rect.right() - m.right()
        for it in self._items:
            w = it.sizeHint().width()
            h = it.sizeHint().height()
            sig = x + w + self._esp
            if sig - self._esp > derecha and alto_fila > 0:
                x = rect.x() + m.left()
                y = y + alto_fila + self._esp
                sig = x + w + self._esp
                alto_fila = 0
            if not solo_prueba:
                it.setGeometry(QRect(QPoint(x, y), it.sizeHint()))
            x = sig
            alto_fila = max(alto_fila, h)
        return y + alto_fila - rect.y() + m.bottom()


# ------------------- Tarjeta de coincidencia -------------------
class Tarjeta(QFrame):
    def __init__(self, ruta, pixmap):
        super().__init__()
        self.ruta = ruta
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(MINI + 16)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        img = QLabel()
        img.setPixmap(pixmap)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap = QLabel(os.path.basename(ruta))
        cap.setObjectName("cap")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setFixedWidth(MINI)
        m = cap.fontMetrics().elidedText(os.path.basename(ruta), Qt.TextElideMode.ElideMiddle, MINI)
        cap.setText(m)
        lay.addWidget(img, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(cap, alignment=Qt.AlignmentFlag.AlignCenter)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            try:
                os.startfile(self.ruta)
            except Exception:
                pass


# ------------------- Ventana principal -------------------
class Ventana(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("◉  FACE FINDER — Buscador de Personas")
        self.resize(1280, 820)          # tamaño al restaurar (se abre maximizada)
        self.setMinimumSize(1000, 660)

        self.motor = None
        self.huella_ref = None
        self.hilo_det = None
        self.hilo_ana = None
        self.hilo_busq = None
        self.carpeta_res = None
        self.encontradas = 0
        self.total_fotos = 0
        self._img_det = None
        self._caras = None

        self._ui()
        self._iniciar_motor()

    # ---------- construcción de la interfaz ----------
    def _ui(self):
        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(18, 16, 18, 16)
        raiz.setSpacing(12)

        # encabezado
        cab = QHBoxLayout()
        marca = QLabel("◉ FACE FINDER")
        marca.setObjectName("brand")
        sub = QLabel("reconocimiento facial local · privado")
        sub.setObjectName("muted")
        cab.addWidget(marca)
        cab.addWidget(sub)
        cab.addStretch()
        raiz.addLayout(cab)

        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(16)
        raiz.addLayout(cuerpo, 1)

        # ---------------- PANEL IZQUIERDO ----------------
        izq = QFrame()
        izq.setObjectName("leftPanel")
        izq.setFixedWidth(520)
        li = QVBoxLayout(izq)
        li.setContentsMargins(18, 18, 18, 18)
        li.setSpacing(10)

        t1 = QLabel("ROSTRO DE REFERENCIA")
        t1.setObjectName("h1")
        t1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s1 = QLabel("malla biométrica en tiempo real")
        s1.setObjectName("muted")
        s1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        li.addWidget(t1)
        li.addWidget(s1)

        self.face = QLabel("◎\n\nCARGA UN ROSTRO")
        self.face.setObjectName("faceFrame")
        self.face.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face.setFixedSize(460, 460)
        self.face.setWordWrap(True)
        li.addWidget(self.face, alignment=Qt.AlignmentFlag.AlignCenter)

        self.btn_rostro = QPushButton("⬆  Cargar rostro")
        self.btn_rostro.setObjectName("primary")
        self.btn_rostro.setEnabled(False)
        self.btn_rostro.clicked.connect(self._cargar_rostro)
        li.addWidget(self.btn_rostro)

        self.estado_ref = QLabel("Cargando modelos…")
        self.estado_ref.setObjectName("muted")
        self.estado_ref.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.estado_ref.setWordWrap(True)
        li.addWidget(self.estado_ref)

        ley = QVBoxLayout()
        ley.setSpacing(3)
        for color, txt in [("#22d3ee", "Mandíbula / mentón"), ("#3b82f6", "Cejas / órbita"),
                           ("#22d3ee", "Ojos"), ("#22d3ee", "Boca")]:
            f = QHBoxLayout()
            pip = QLabel()
            pip.setFixedSize(22, 4)
            pip.setStyleSheet(f"background:{color}; border-radius:2px;")
            lab = QLabel("  " + txt)
            lab.setObjectName("legend")
            f.addWidget(pip)
            f.addWidget(lab)
            f.addStretch()
            ley.addLayout(f)
        li.addLayout(ley)
        li.addStretch()
        cuerpo.addWidget(izq)

        # ---------------- PANEL DERECHO ----------------
        der = QVBoxLayout()
        der.setSpacing(10)
        cuerpo.addLayout(der, 1)

        fila_t = QHBoxLayout()
        t2 = QLabel("RECONOCIMIENTO")
        t2.setObjectName("h1")
        self.btn_reiniciar = QPushButton("↻  Reiniciar")
        self.btn_reiniciar.setObjectName("ghost")
        self.btn_reiniciar.clicked.connect(self._reiniciar)
        fila_t.addWidget(t2)
        fila_t.addStretch()
        fila_t.addWidget(self.btn_reiniciar)
        der.addLayout(fila_t)

        # carpeta
        fila = QHBoxLayout()
        self.entrada = QLineEdit()
        self.entrada.setReadOnly(True)
        self.entrada.setPlaceholderText("Ninguna carpeta seleccionada…")
        btn_car = QPushButton("Elegir carpeta")
        btn_car.setObjectName("ghost")
        btn_car.clicked.connect(self._elegir_carpeta)
        fila.addWidget(self.entrada, 1)
        fila.addWidget(btn_car)
        der.addLayout(fila)

        # precisión + copiar
        fila2 = QHBoxLayout()
        etq = QLabel("Precisión:")
        etq.setObjectName("muted")
        fila2.addWidget(etq)
        self.grupo = QButtonGroup(self)
        for i, nombre in enumerate(UMBRALES):
            rb = QRadioButton(nombre)
            if nombre == "Normal":
                rb.setChecked(True)
            self.grupo.addButton(rb, i)
            fila2.addWidget(rb)
        fila2.addStretch()
        self.chk_copiar = QCheckBox("Copiar a carpeta")
        self.chk_copiar.setChecked(True)
        fila2.addWidget(self.chk_copiar)
        der.addLayout(fila2)

        # botones iniciar / detener
        fila3 = QHBoxLayout()
        self.btn_buscar = QPushButton("▶  Iniciar reconocimiento")
        self.btn_buscar.setObjectName("primary")
        self.btn_buscar.setEnabled(False)
        self.btn_buscar.clicked.connect(self._iniciar_busqueda)
        self.btn_detener = QPushButton("■  Detener")
        self.btn_detener.setObjectName("danger")
        self.btn_detener.setEnabled(False)
        self.btn_detener.clicked.connect(self._detener)
        fila3.addWidget(self.btn_buscar)
        fila3.addWidget(self.btn_detener)
        fila3.addStretch()
        der.addLayout(fila3)

        self.barra = QProgressBar()
        self.barra.setValue(0)
        self.barra.setTextVisible(False)
        der.addWidget(self.barra)

        self.estado = QLabel("Carga un rostro a la izquierda para empezar.")
        self.estado.setObjectName("muted")
        der.addWidget(self.estado)

        # cabecera carrusel
        fila4 = QHBoxLayout()
        self.titulo_c = QLabel("COINCIDENCIAS")
        self.titulo_c.setObjectName("h1")
        pista = QLabel("clic para abrir")
        pista.setObjectName("muted")
        self.btn_carpeta = QPushButton("📂 Carpeta")
        self.btn_carpeta.setObjectName("ghost")
        self.btn_carpeta.setEnabled(False)
        self.btn_carpeta.clicked.connect(self._abrir_carpeta_res)
        fila4.addWidget(self.titulo_c)
        fila4.addWidget(pista)
        fila4.addStretch()
        fila4.addWidget(self.btn_carpeta)
        der.addLayout(fila4)

        # área de scroll con FlowLayout
        self.scroll = QScrollArea()
        self.scroll.setObjectName("scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cont = QWidget()
        self.cont.setObjectName("flowHost")
        self.flow = FlowLayout(self.cont, margen=12, espacio=12)
        self.scroll.setWidget(self.cont)
        der.addWidget(self.scroll, 1)

        self._placeholder("Aún no hay coincidencias.")

    # ---------- utilidades ----------
    def _placeholder(self, texto):
        self._limpiar_flow()
        lab = QLabel("   " + texto + "   ")
        lab.setObjectName("muted")
        self.flow.addWidget(lab)

    def _limpiar_flow(self):
        while self.flow.count():
            it = self.flow.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

    # ---------- arranque del motor ----------
    def _iniciar_motor(self):
        self.hilo_init = HiloInit()
        self.hilo_init.listo.connect(self._motor_listo)
        self.hilo_init.start()

    def _motor_listo(self, motor):
        self.motor = motor
        self.btn_rostro.setEnabled(True)
        self.estado_ref.setText("Listo. Carga una foto del rostro.")

    # ---------- panel izquierdo ----------
    def _cargar_rostro(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Elige una foto clara y de frente del rostro", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff "
            "*.heic *.heif *.avif)")
        if not ruta:
            return
        self.face.setText("◌\n\nDETECTANDO…")
        self.face.setPixmap(QPixmap())
        self.estado_ref.setText("Detectando rostros…")
        self.estado_ref.setStyleSheet("")
        self.btn_rostro.setEnabled(False)
        self.btn_buscar.setEnabled(False)
        self.hilo_det = HiloDetectar(self.motor, ruta)
        self.hilo_det.listo.connect(self._rostros_detectados)
        self.hilo_det.error.connect(self._rostro_error)
        self.hilo_det.start()

    def _rostros_detectados(self, img_det, caras, recortes):
        self._img_det = img_det
        self._caras = caras
        if len(caras) == 1:
            self._analizar_idx(0)
            return
        dlg = DialogoRostros(recortes, self)
        dlg.exec()
        if dlg.idx >= 0:
            self._analizar_idx(dlg.idx)
        else:
            self.face.setText("◎\n\nCARGA UN ROSTRO")
            self.face.setPixmap(QPixmap())
            self.estado_ref.setText("Selección cancelada.")
            self.btn_rostro.setEnabled(True)

    def _analizar_idx(self, idx):
        self.face.setText("◌\n\nANALIZANDO…")
        self.estado_ref.setText("Analizando rostro…")
        self.btn_rostro.setEnabled(False)
        self.hilo_ana = HiloAnalizar(self.motor, self._img_det, self._caras, idx)
        self.hilo_ana.listo.connect(self._rostro_listo)
        self.hilo_ana.error.connect(self._rostro_error)
        self.hilo_ana.start()

    def _rostro_listo(self, huella, mesh):
        self.huella_ref = huella
        pix = bgr_a_pixmap(mesh).scaled(
            456, 456, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.face.setPixmap(pix)
        self.estado_ref.setText("✓ Rostro detectado · malla lista")
        self.estado_ref.setStyleSheet("color:#34d399;")
        self.btn_rostro.setEnabled(True)
        self.btn_buscar.setEnabled(True)
        self.estado.setText("Elige la carpeta y pulsa Iniciar reconocimiento.")

    def _rostro_error(self, msg):
        self.huella_ref = None
        self.face.setText("⚠\n\nSIN ROSTRO")
        self.face.setPixmap(QPixmap())
        self.estado_ref.setText(msg)
        self.estado_ref.setStyleSheet("color:#f87171;")
        self.btn_rostro.setEnabled(True)

    # ---------- panel derecho ----------
    def _elegir_carpeta(self):
        ruta = QFileDialog.getExistingDirectory(self, "Elige la carpeta con todas tus fotos")
        if ruta:
            self.entrada.setText(ruta)

    def _iniciar_busqueda(self):
        if self.huella_ref is None:
            QMessageBox.warning(self, "Falta el rostro", "Primero carga un rostro válido.")
            return
        carpeta = self.entrada.text()
        if not carpeta or not os.path.isdir(carpeta):
            QMessageBox.warning(self, "Falta la carpeta", "Elige la carpeta con tus fotos.")
            return

        umbral = UMBRALES[list(UMBRALES)[self.grupo.checkedId()]]
        self.encontradas = 0
        self.titulo_c.setText("COINCIDENCIAS")
        self._limpiar_flow()
        self.barra.setValue(0)
        self.btn_buscar.setEnabled(False)
        self.btn_detener.setEnabled(True)
        self.btn_carpeta.setEnabled(False)
        self.estado.setText("Preparando…")

        self.hilo_busq = HiloBusqueda(self.motor, self.huella_ref, carpeta,
                                      umbral, self.chk_copiar.isChecked())
        self.hilo_busq.total.connect(self._on_total)
        self.hilo_busq.progreso.connect(self._on_progreso)
        self.hilo_busq.encontrada.connect(self._on_encontrada)
        self.hilo_busq.terminado.connect(self._on_terminado)
        self.hilo_busq.error.connect(self._on_error)
        self.hilo_busq.start()

    def _detener(self):
        if self.hilo_busq:
            self.hilo_busq.cancelar()
        self.btn_detener.setEnabled(False)
        self.estado.setText("Deteniendo la búsqueda…")

    def _on_total(self, total):
        self.total_fotos = total
        self.barra.setMaximum(total)
        self.estado.setText(f"Analizando {total} fotos…")

    def _on_progreso(self, i, nombre):
        self.barra.setValue(i)
        self.estado.setText(f"Analizando {i}/{self.total_fotos}  ·  "
                            f"{self.encontradas} encontradas  ·  {nombre[:26]}")

    def _on_encontrada(self, ruta):
        if self.encontradas == 0:
            self._limpiar_flow()
        pix = miniatura_pixmap(ruta)
        if pix is None:
            return
        self.encontradas += 1
        self.flow.addWidget(Tarjeta(ruta, pix))
        self.titulo_c.setText(f"COINCIDENCIAS ({self.encontradas})")

    def _on_terminado(self, motivo):
        self.carpeta_res = getattr(self.hilo_busq, "destino", None)
        self._reset_botones()
        if self.carpeta_res and self.encontradas > 0:
            self.btn_carpeta.setEnabled(True)
        if motivo == "cancelado":
            self.estado.setText(f"⏹ Búsqueda detenida · {self.encontradas} encontradas.")
        else:
            self.estado.setText(f"✓ Listo · {self.encontradas} foto(s) con esa persona.")
        if self.encontradas == 0 and motivo != "cancelado":
            self._placeholder("No hubo coincidencias. Prueba precisión 'Amplia'.")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Aviso", msg)
        self._reset_botones()
        self.estado.setText("Detenido.")

    def _reset_botones(self):
        self.btn_buscar.setEnabled(True)
        self.btn_detener.setEnabled(False)

    def _abrir_carpeta_res(self):
        if self.carpeta_res and os.path.isdir(self.carpeta_res):
            os.startfile(self.carpeta_res)

    def _reiniciar(self):
        """Deja todo como al abrir el programa, listo para un nuevo análisis."""
        if self.hilo_busq and self.hilo_busq.isRunning():
            self.hilo_busq.cancelar()
            self.hilo_busq.wait(1500)
        self.huella_ref = None
        self._img_det = None
        self._caras = None
        self.encontradas = 0
        self.total_fotos = 0
        self.carpeta_res = None
        self.face.setText("◎\n\nCARGA UN ROSTRO")
        self.face.setPixmap(QPixmap())
        self.estado_ref.setText("Listo. Carga una foto del rostro.")
        self.estado_ref.setStyleSheet("")
        self.entrada.clear()
        self.barra.setValue(0)
        self.titulo_c.setText("COINCIDENCIAS")
        self._placeholder("Aún no hay coincidencias.")
        self.btn_rostro.setEnabled(self.motor is not None)
        self.btn_buscar.setEnabled(False)
        self.btn_detener.setEnabled(False)
        self.btn_carpeta.setEnabled(False)
        self.estado.setText("Carga un rostro a la izquierda para empezar.")

    def closeEvent(self, e):
        for h in (self.hilo_busq, self.hilo_det, self.hilo_ana):
            if h and h.isRunning():
                if hasattr(h, "cancelar"):
                    h.cancelar()
                h.wait(1500)
        e.accept()


ESTILO = """
* { font-family: 'Segoe UI'; }
QWidget { background: #060a12; color: #dbeafe; font-size: 12px; }
QLabel { background: transparent; }
QLabel#brand { color: #22d3ee; font-size: 20px; font-weight: 800; letter-spacing: 3px; }
QLabel#h1 { color: #dbeafe; font-size: 14px; font-weight: 700; letter-spacing: 1px; }
QLabel#muted { color: #5f7899; font-size: 11px; }
QLabel#legend { color: #dbeafe; font-size: 11px; }
#leftPanel { background: #0c1424; border: 1px solid #1f3153; border-radius: 16px; }
#faceFrame { background: #02060c; border: 1px solid #22d3ee; border-radius: 12px; color: #395268; font-size: 14px; }
QLineEdit { background: #111c30; border: 1px solid #1f3153; border-radius: 8px; padding: 7px 10px; color: #dbeafe; }
QLineEdit:focus { border: 1px solid #22d3ee; }
QPushButton#primary {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #22d3ee, stop:1 #0ea5c4);
    color: #03212b; border: none; border-radius: 8px; padding: 9px 18px; font-weight: 700;
}
QPushButton#primary:hover { background: #3fe0f2; }
QPushButton#primary:disabled { background: #16243c; color: #46607d; }
QPushButton#danger { background: #ef4444; color: white; border: none; border-radius: 8px; padding: 9px 18px; font-weight: 700; }
QPushButton#danger:hover { background: #f26363; }
QPushButton#danger:disabled { background: #16243c; color: #46607d; }
QPushButton#ghost { background: #111c30; color: #dbeafe; border: 1px solid #1f3153; border-radius: 8px; padding: 8px 14px; }
QPushButton#ghost:hover { border: 1px solid #22d3ee; }
QPushButton#ghost:disabled { color: #46607d; border-color: #16243c; }
QProgressBar { background: #111c30; border: none; border-radius: 6px; min-height: 10px; max-height: 10px; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3b82f6, stop:1 #22d3ee); border-radius: 6px; }
QRadioButton, QCheckBox { color: #dbeafe; spacing: 6px; }
QRadioButton::indicator, QCheckBox::indicator { width: 14px; height: 14px; }
QRadioButton::indicator { border: 1px solid #2b3f63; border-radius: 8px; background: #111c30; }
QRadioButton::indicator:checked { border: 4px solid #22d3ee; background: #02060c; }
QCheckBox::indicator { border: 1px solid #2b3f63; border-radius: 4px; background: #111c30; }
QCheckBox::indicator:checked { border: 1px solid #22d3ee; background: #22d3ee; }
#scroll { background: #0c1424; border: 1px solid #1f3153; border-radius: 12px; }
#flowHost { background: transparent; }
QScrollArea { border: 1px solid #1f3153; border-radius: 12px; }
#card { background: #111c30; border: 1px solid #1f3153; border-radius: 10px; }
#card:hover { border: 1px solid #22d3ee; }
QLabel#cap { color: #5f7899; font-size: 9px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 4px; }
QScrollBar::handle:vertical { background: #1f3153; border-radius: 5px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #22d3ee; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QMessageBox { background: #0c1424; }
QDialog { background: #0c1424; }
QToolButton#facechoice { background: #111c30; border: 1px solid #1f3153; border-radius: 10px; padding: 8px; color: #dbeafe; }
QToolButton#facechoice:hover { border: 1px solid #22d3ee; }
"""


def main():
    if any(not os.path.isfile(m) for m in (MODELO_DET, MODELO_REC)):
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Faltan los modelos",
                             "No encontré los modelos en 'modelos'. Descárgalos según el README.")
        return
    app = QApplication(sys.argv)
    app.setStyleSheet(ESTILO)
    v = Ventana()
    v.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
