"""
تكبير الصور باستيفاء لاغرانج | Lagrange Image Resizer

لكل بكسل جديد نحسب موقعه داخل الصورة الأصلية ونأخذ أقرب N بكسل كعقد،
ثم نطبّق متعددة حدود لاغرانج:
    y = sum( y_j * L_j(u) )   حيث   L_j(u) = prod( (u - x_m) / (x_j - x_m) ) لكل m != j

- N = 2 : استيفاء خطي (Linear)
- N = 4 : استيفاء تكعيبي (Cubic) — الافتراضي
- N = 6 : استيفاء خماسي (Quintic)

بما أن الصورة شبكة ثنائية الأبعاد، نطبّق الاستيفاء أفقياً ثم عمودياً على دفعات لتوفير الذاكرة.

الاستخدام:
    python lagrange_image_resizer.py [صورة]
    python lagrange_image_resizer.py input.png -o output.png -s 2 -p 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Callable

# ---------------------------------------------------------------- التحقق من المكتبات


def ensure_dependencies() -> None:
    """تثبيت المكتبات الأساسية تلقائياً عند التشغيل التفاعلي إذا كانت ناقصة."""
    missing = []
    for module_name, package_name in (("numpy", "numpy"), ("PIL", "Pillow")):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if not missing:
        return

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except (subprocess.CalledProcessError, OSError):
        message = (
            "تعذر تثبيت المكتبات المطلوبة تلقائياً:\n"
            f"Missing packages: {', '.join(missing)}\n\n"
            f"يرجى تثبيتها يدوياً عبر الأمر:\npip install {' '.join(missing)}"
        )
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("مكتبات ناقصة | Missing Packages", message)
        except Exception:
            print(message, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    ensure_dependencies()

import numpy as np  # noqa: E402
from PIL import Image, ImageOps, ImageTk  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

try:
    RESAMPLE_DOWN = Image.Resampling.LANCZOS  # Pillow 9.1+
    RESAMPLE_UP = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_DOWN = Image.LANCZOS
    RESAMPLE_UP = Image.NEAREST


# ---------------------------------------------------------------- ثوابت الحساب

POINTS_CHOICES = (2, 4, 6)       # خطي، تكعيبي، خماسي
MIN_SCALE, MAX_SCALE = 1.0, 8.0
CONFIRM_MEGAPIXELS = 40          # تنبيه المستخدم قبل تجاوز هذا الحجم
MAX_MEGAPIXELS = 120             # الحد الأقصى المسموح به لحماية الذاكرة
BAND_ELEMENTS = 6_000_000        # عدد العناصر في الدفعة الواحدة أثناء المعالجة


class Cancelled(Exception):
    """يُرفع عند إلغاء المستخدم لعملية التكبير أثناء التنفيذ."""


# ---------------------------------------------------------------- نواة استيفاء لاغرانج


def lagrange_weights(
    in_len: int, out_len: int, points: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """حساب أوزان لاغرانج ومؤشرات العقد لمحور واحد بالشكل (out_len, points)."""
    if points < 2 or points % 2 != 0:
        raise ValueError("يجب أن يكون عدد النقاط عدداً زوجياً أكبر من أو يساوي 2")

    # محاذاة مراكز البكسلات بين الشبكة الأصلية والشبكة الجديدة
    src_pos = (np.arange(out_len, dtype=np.float64) + 0.5) * (in_len / out_len) - 0.5
    anchor = np.floor(src_pos).astype(np.int64)
    offset = src_pos - anchor  # الإزاحة الكسرية ضمن [0, 1)

    half = points // 2
    nodes = np.arange(-half + 1, half + 1, dtype=np.int64)  # مثلاً: [-1, 0, 1, 2]
    weights = np.ones((out_len, points), dtype=np.float64)

    for j in range(points):
        for m in range(points):
            if m != j:
                weights[:, j] *= (offset - nodes[m]) / (nodes[j] - nodes[m])

    # عند حواف الصورة نكرر البكسل الطرفي مع بقاء الأوزان محسوبة على المواقع الحقيقية
    indices = np.clip(anchor[:, None] + nodes[None, :], 0, in_len - 1).astype(np.int64)
    return weights.astype(np.float32), indices


def lagrange_resize(
    img: np.ndarray,
    new_w: int,
    new_h: int,
    points: int = 4,
    progress: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    """تكبير مصفوفة صورة uint8 بالشكل (H, W, 3 أو 4) باستخدام استيفاء لاغرانج القابل للفصل."""
    src_h, src_w, channels = img.shape
    weights_x, indices_x = lagrange_weights(src_w, new_w, points)
    weights_y, indices_y = lagrange_weights(src_h, new_h, points)

    weights_x_T = np.ascontiguousarray(weights_x.T)
    indices_x_T = np.ascontiguousarray(indices_x.T)
    weights_y_T = np.ascontiguousarray(weights_y.T)
    has_alpha = channels == 4

    output = np.empty((new_h, new_w, channels), dtype=np.uint8)
    rows_per_band = max(1, min(new_h, BAND_ELEMENTS // (new_w * channels)))

    for row_start in range(0, new_h, rows_per_band):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()

        row_end = min(new_h, row_start + rows_per_band)
        band_rows = indices_y[row_start:row_end]
        src_top = int(band_rows.min())
        src_bottom = int(band_rows.max())

        src_patch = img[src_top : src_bottom + 1].astype(np.float32)
        if has_alpha:
            # ضرب الألوان بالشفافية مسبقاً لمنع ظهور هالات داكنة عند الحواف الشفافة
            src_patch[..., :3] *= src_patch[..., 3:4] / np.float32(255.0)

        # 1) التمرير الأفقي مع إعادة استخدام المصفوفة المؤقتة لتقليل استهلاك الذاكرة
        horiz = np.zeros((src_patch.shape[0], new_w, channels), dtype=np.float32)
        horiz_scratch = np.empty_like(horiz)
        for k in range(points):
            np.take(src_patch, indices_x_T[k], axis=1, out=horiz_scratch)
            horiz_scratch *= weights_x_T[k][None, :, None]
            horiz += horiz_scratch

        # 2) التمرير العمودي
        band_h = row_end - row_start
        vert = np.zeros((band_h, new_w, channels), dtype=np.float32)
        vert_scratch = np.empty_like(vert)
        for k in range(points):
            np.take(horiz, band_rows[:, k] - src_top, axis=0, out=vert_scratch)
            vert_scratch *= weights_y_T[k, row_start:row_end][:, None, None]
            vert += vert_scratch

        if has_alpha:
            alpha = np.clip(vert[..., 3:4], np.float32(0.0), np.float32(255.0))
            inv_alpha = np.where(
                alpha > np.float32(0.5),
                np.float32(255.0) / np.maximum(alpha, np.float32(1e-6)),
                np.float32(0.0),
            ).astype(np.float32)
            vert[..., :3] *= inv_alpha
            vert[..., 3:4] = alpha

        # التقريب والقص مرة واحدة في نهاية الدفعة للحفاظ على الدقة
        np.rint(vert, out=vert)
        np.clip(vert, np.float32(0.0), np.float32(255.0), out=vert)
        output[row_start:row_end] = vert.astype(np.uint8)

        if progress is not None:
            progress(row_end / new_h)

    return output


# ---------------------------------------------------------------- قراءة وحفظ الصور


def load_image_array(path: str | os.PathLike) -> tuple[np.ndarray, bool, bytes | None]:
    """تحميل الصورة كمصفوفة RGB أو RGBA مع تصحيح اتجاه EXIF وحفظ ملف الألوان ICC."""
    with Image.open(path) as im:
        icc_profile = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info

        if has_alpha:
            im = im.convert("RGBA")
            # إذا كانت قناة الشفافية معتمة بالكامل نحولها إلى RGB لتسريع المعالجة
            if im.getchannel("A").getextrema()[0] == 255:
                im = im.convert("RGB")
                has_alpha = False
        else:
            im = im.convert("RGB")

        return np.array(im, dtype=np.uint8), has_alpha, icc_profile


def save_image_array(arr: np.ndarray, path: str | os.PathLike, icc: bytes | None = None) -> None:
    """حفظ المصفوفة كملف صورة بالصيغة المناسبة مع دعم الشفافية وملف الألوان ICC."""
    ext = os.path.splitext(str(path))[1].lower()
    img = Image.fromarray(arr)

    # صيغ JPEG و BMP لا تدعم الشفافية، فنضع الصورة فوق خلفية بيضاء
    if img.mode == "RGBA" and ext in (".jpg", ".jpeg", ".bmp"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background

    save_kwargs: dict[str, object] = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs.update(quality=95, subsampling=0)
    elif ext == ".webp":
        save_kwargs.update(quality=95)

    if icc and ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
        save_kwargs["icc_profile"] = icc

    img.save(path, **save_kwargs)


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")


def parse_scale(text: str) -> float:
    """تحويل نص معامل التكبير (مثل '2' أو '2x' أو '×2.5' أو '٣') إلى عدد عشري."""
    cleaned = text.strip().lower().translate(_ARABIC_DIGITS)
    cleaned = cleaned.replace("×", "").replace("x", "").strip()
    value = float(cleaned)
    if not (MIN_SCALE < value <= MAX_SCALE):
        raise ValueError("معامل التكبير خارج النطاق المسموح")
    return value


# ---------------------------------------------------------------- النصوص والألوان

TEXTS = {
    "rtl": {
        "win_title": "تكبير الصور باستيفاء لاغرانج | Lagrange Image Resizer",
        "header": "‏نظام تكبير الصور وتحسين الدقة (Lagrange Interpolation)",
        "lang_btn": "🌐 الواجهة: RTL (عربي) ⇄ LTR",
        "load_btn": "📂 1. اختيار صورة",
        "scale_lbl": "‏2. معامل التكبير:",
        "method_lbl": "‏رتبة الاستيفاء:",
        "methods": ["خطي (نقطتان)", "تكعيبي (4 نقاط) — افتراضي", "خماسي (6 نقاط)"],
        "run_btn": "⚡ 3. تكبير بمعادلة لاغرانج",
        "cancel_btn": "⛔ إلغاء العملية",
        "save_btn": "💾 4. حفظ الصورة المكبرة",
        "view_btn": "🔍 عرض بحجم 100% (تفاصيل)",
        "orig_title": "‏الصورة الأصلية (Original)",
        "res_title": "‏الصورة المكبرة (Lagrange Output)",
        "orig_empty": "‏لم يتم اختيار صورة بعد\nاضغط على (اختيار صورة)\n‏(Ctrl+O)",
        "res_empty": "‏النتيجة ستظهر هنا بعد الضغط على\n(تكبير بمعادلة لاغرانج)",
        "status_ready": "‏الحالة: البرنامج جاهز — اختر صورة من حاسوبك للبدء.",
        "status_loaded": "‏تم تحميل الصورة | الأبعاد الأصلية: ‎{w} × {h} px‏ | ‎{name}",
        "status_working": "‏جارٍ التكبير… ‎{p}%",
        "status_done": (
            "‏تم التكبير بمعامل (‎{s}x‏) بمعادلة لاغرانج "
            "(‎{n}‏ نقاط) | الأبعاد الجديدة: ‎{w} × {h} px‏"
            " | الزمن: ‎{sec:.1f}s"
        ),
        "status_cancelled": "‏تم إلغاء العملية.",
        "status_saved": "‏تم حفظ الصورة المكبرة بنجاح: ‎{path}",
        "status_error": "‏حدث خطأ — {msg}",
        "dim_badge": "‏الأبعاد الحقيقية: ‎{w} × {h} px",
        "dim_expected": "‏الأبعاد المتوقعة: ‎{w} × {h} px",
        "warn_title": "تنبيه",
        "warn_no_img": "‏يرجى اختيار صورة من جهازك أولاً!",
        "err_title": "خطأ",
        "err_open": "‏تعذر فتح الصورة:\n{msg}",
        "err_scale": "‏معامل التكبير غير صالح.\nأدخل رقماً أكبر من 1 وحتى 8 (مثل 2 أو 2.5).",
        "err_too_big": (
            "‏الصورة الناتجة كبيرة جداً (‎{mp:.0f}‏ ميغابكسل).\n"
            "‏الحد الأقصى ‎{max}‏ ميغابكسل — اختر معاملاً أصغر."
        ),
        "confirm_big": (
            "‏الصورة الناتجة ستكون ‎{w} × {h} px‏ (≈‎{mp:.0f}‏ ميغابكسل) "
            "وقد تستغرق وقتاً وذاكرة كبيرة.\nهل تريد المتابعة؟"
        ),
        "err_run": "‏فشلت عملية التكبير:\n{msg}",
        "err_memory": "الذاكرة غير كافية — جرّب معاملاً أصغر.",
        "err_save": "‏تعذر حفظ الصورة:\n{msg}",
        "viewer_hint": "اسحب بالماوس أو استخدم الأسهم للتنقل، وزر Home للتوسيط — 1 بكسل = 1 بكسل على الشاشة",
        "open_title": "اختر صورة",
        "save_title": "حفظ الصورة المكبرة",
    },
    "ltr": {
        "win_title": "Lagrange Interpolation - Image Resizer",
        "header": "Lagrange Interpolation Image Resizer & Enhancer",
        "lang_btn": "🌐 Layout: LTR (English) ⇄ RTL",
        "load_btn": "📂 1. Load Image",
        "scale_lbl": "2. Scale Factor:",
        "method_lbl": "Interpolation:",
        "methods": ["Linear (2-point)", "Cubic (4-point) — default", "Quintic (6-point)"],
        "run_btn": "⚡ 3. Apply Lagrange Resize",
        "cancel_btn": "⛔ Cancel",
        "save_btn": "💾 4. Save Resized Image",
        "view_btn": "🔍 View at 100% (detail)",
        "orig_title": "Original Image (Input)",
        "res_title": "Resized Image (Lagrange Output)",
        "orig_empty": "No image loaded yet\nClick 'Load Image'\n(Ctrl+O)",
        "res_empty": "Output will appear here after\napplying Lagrange Interpolation",
        "status_ready": "Status: Ready — Please select an image from your computer.",
        "status_loaded": "Image loaded | Original size: {w} × {h} px | {name}",
        "status_working": "Resizing… {p}%",
        "status_done": (
            "Lagrange interpolation ({s}x, {n}-point) complete | "
            "New size: {w} × {h} px | Time: {sec:.1f}s"
        ),
        "status_cancelled": "Operation cancelled.",
        "status_saved": "Resized image saved successfully: {path}",
        "status_error": "Error — {msg}",
        "dim_badge": "Actual dimensions: {w} × {h} px",
        "dim_expected": "Expected dimensions: {w} × {h} px",
        "warn_title": "Warning",
        "warn_no_img": "Please load an image from your computer first!",
        "err_title": "Error",
        "err_open": "Could not open the image:\n{msg}",
        "err_scale": "Invalid scale factor.\nEnter a number above 1 and up to 8 (e.g. 2 or 2.5).",
        "err_too_big": (
            "The output would be too large ({mp:.0f} MP).\n"
            "The limit is {max} MP — choose a smaller scale."
        ),
        "confirm_big": (
            "The output will be {w} × {h} px (≈{mp:.0f} MP) and may take a while "
            "and use a lot of memory.\nContinue?"
        ),
        "err_run": "Resize failed:\n{msg}",
        "err_memory": "Not enough memory — try a smaller scale.",
        "err_save": "Could not save the image:\n{msg}",
        "viewer_hint": "Drag, use arrow keys or wheel to pan, Home to center — 1 image pixel = 1 screen pixel",
        "open_title": "Select Image",
        "save_title": "Save Resized Image",
    },
}

# لوحة الألوان (Catppuccin Mocha)
APP_BG, HEADER_BG, CARD_BG, PREVIEW_BG, STATUS_BG = "#1E1E2E", "#181825", "#28283D", "#181825", "#11111B"
TEXT, SUBTLE, MUTED = "#CDD6F4", "#A6ADC8", "#6C7086"
BLUE, GREEN, YELLOW = "#89B4FA", "#A6E3A1", "#F9E2AF"
CTRL_BG, CTRL_ACTIVE = "#313244", "#45475A"
PREVIEW_CACHE_PX = 1600


@dataclass
class PreviewPanel:
    """تجميع عناصر بطاقة المعاينة الواحدة (الأصلية أو المكبّرة)."""

    empty_key: str
    card: tk.Frame
    title: tk.Label
    holder: tk.Frame
    view: tk.Label
    dim: tk.Label
    view_btn: tk.Button
    cached_thumb: Image.Image | None = None
    photo: ImageTk.PhotoImage | None = None
    resize_after_id: str | None = None


# ---------------------------------------------------------------- نافذة الفحص بحجم 100%


class PanViewer(tk.Toplevel):
    """عارض تفاصيل بالحجم الحقيقي (1:1) يرسم الجزء المرئي فقط لتصفح الصور الضخمة بسلاسة."""

    def __init__(self, master: tk.Misc, arr: np.ndarray, title: str, hint: str) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("920x680")
        self.minsize(400, 300)
        self.configure(bg=STATUS_BG)

        self.arr = arr
        h, w = arr.shape[:2]
        self.offset_x = max(0, (w - 920) // 2)
        self.offset_y = max(0, (h - 650) // 2)
        self.photo: ImageTk.PhotoImage | None = None
        self._drag_origin: tuple[int, int, int, int] | None = None

        self.canvas = tk.Canvas(self, bg=STATUS_BG, highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)

        tk.Label(
            self,
            text=f"{w} × {h} px   |   {hint}",
            bg=HEADER_BG,
            fg=SUBTLE,
            font=("Segoe UI", 9),
            pady=5,
        ).pack(fill="x")

        self.canvas.bind("<Configure>", lambda _e: self.render())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(seq, self._on_wheel)

        for key, dx, dy in (("<Left>", -80, 0), ("<Right>", 80, 0), ("<Up>", 0, -80), ("<Down>", 0, 80)):
            self.bind(key, lambda _e, dx=dx, dy=dy: self.pan(dx, dy))

        self.bind("<Home>", lambda _e: self.center_view())
        self.bind("r", lambda _e: self.center_view())
        self.bind("R", lambda _e: self.center_view())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.canvas.focus_set()

    def center_view(self) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        h, w = self.arr.shape[:2]
        self.offset_x = max(0, (w - cw) // 2)
        self.offset_y = max(0, (h - ch) // 2)
        self.render()

    def pan(self, dx: int, dy: int) -> None:
        self.offset_x += dx
        self.offset_y += dy
        self.render()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_origin = (event.x, event.y, self.offset_x, self.offset_y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_origin is None:
            return
        start_x, start_y, orig_ox, orig_oy = self._drag_origin
        self.offset_x = orig_ox - (event.x - start_x)
        self.offset_y = orig_oy - (event.y - start_y)
        self.render()

    def _on_wheel(self, event: tk.Event) -> None:
        scroll_up = getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0
        step = -150 if scroll_up else 150
        if event.state & 0x1:  # مع Shift نمرر أفقياً
            self.pan(step, 0)
        else:
            self.pan(0, step)

    def render(self) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        h, w = self.arr.shape[:2]
        if cw < 2 or ch < 2:
            return

        self.offset_x = int(min(max(self.offset_x, 0), max(0, w - cw)))
        self.offset_y = int(min(max(self.offset_y, 0), max(0, h - ch)))

        crop = np.ascontiguousarray(
            self.arr[self.offset_y : self.offset_y + ch, self.offset_x : self.offset_x + cw]
        )
        self.photo = ImageTk.PhotoImage(Image.fromarray(crop))
        self.canvas.delete("all")
        self.canvas.create_image(
            max(0, (cw - crop.shape[1]) // 2),
            max(0, (ch - crop.shape[0]) // 2),
            anchor="nw",
            image=self.photo,
        )


# ---------------------------------------------------------------- التطبيق الرئيسي


class LagrangeImageResizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.geometry("1100x760")
        root.minsize(1020, 680)
        root.configure(bg=APP_BG)

        self.is_rtl = True
        self.scale = 2.0
        self.orig_arr: np.ndarray | None = None
        self.res_arr: np.ndarray | None = None
        self.has_alpha = False
        self.icc: bytes | None = None
        self.src_path = ""

        self.busy = False
        self.cancel_event = threading.Event()
        self.result_queue: queue.Queue[tuple[str, object]] | None = None
        self.start_time = 0.0
        self.job_points = 4
        self.status: tuple[str, dict[str, object]] = ("status_ready", {})

        self.setup_styles()
        self.build_ui()
        self.bind_shortcuts()
        self.apply_direction_layout()

    def t(self) -> dict[str, object]:
        return TEXTS["rtl" if self.is_rtl else "ltr"]

    def set_status(self, key: str, **kwargs: object) -> None:
        self.status = (key, kwargs)
        self.render_status()

    def render_status(self) -> None:
        key, kwargs = self.status
        template = str(self.t()[key])
        self.status_bar.config(text=template.format(**kwargs))

    def setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Lagrange.TCombobox",
            fieldbackground=CTRL_BG,
            background=CTRL_ACTIVE,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor=CTRL_BG,
            lightcolor=CTRL_BG,
            darkcolor=CTRL_BG,
            selectbackground=CTRL_BG,
            selectforeground=TEXT,
        )
        style.map(
            "Lagrange.TCombobox",
            fieldbackground=[("readonly", CTRL_BG), ("disabled", CARD_BG)],
            foreground=[("disabled", MUTED)],
            selectbackground=[("readonly", CTRL_BG)],
            selectforeground=[("readonly", TEXT)],
        )
        style.configure(
            "Lagrange.Horizontal.TProgressbar",
            troughcolor=HEADER_BG,
            background=GREEN,
            bordercolor=HEADER_BG,
            lightcolor=GREEN,
            darkcolor=GREEN,
            thickness=6,
        )
        self.root.option_add("*TCombobox*Listbox.background", CTRL_BG)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BLUE)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#11111B")

    def build_ui(self) -> None:
        # الشريط العلوي
        self.header_frame = tk.Frame(self.root, bg=HEADER_BG, pady=14, padx=20)
        self.header_frame.pack(fill="x")
        self.title_lbl = tk.Label(self.header_frame, bg=HEADER_BG, fg=TEXT, font=("Segoe UI", 14, "bold"))
        self.lang_btn = tk.Button(
            self.header_frame,
            bg=CTRL_BG,
            fg=BLUE,
            activebackground=CTRL_ACTIVE,
            activeforeground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.toggle_direction,
        )

        # شريط الحالة وشريط التقدم في الأسفل
        self.status_bar = tk.Label(self.root, bg=STATUS_BG, fg=TEXT, font=("Segoe UI", 10), padx=20, pady=8)
        self.status_bar.pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(
            self.root,
            mode="determinate",
            maximum=100,
            style="Lagrange.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", side="bottom")

        # صف الإعدادات وصف أزرار التنفيذ والحفظ
        self.ctrl_frame = tk.Frame(self.root, bg=APP_BG, pady=10, padx=20)
        self.ctrl_frame.pack(fill="x")
        self.action_frame = tk.Frame(self.root, bg=APP_BG, pady=0, padx=20)
        self.action_frame.pack(fill="x", pady=(0, 6))

        btn_style = {
            "font": ("Segoe UI", 10, "bold"),
            "bd": 0,
            "padx": 16,
            "pady": 8,
            "cursor": "hand2",
            "disabledforeground": MUTED,
        }

        self.load_btn = tk.Button(
            self.ctrl_frame,
            bg=BLUE,
            fg="#11111B",
            activebackground="#B4BEFE",
            command=self.open_image_file,
            **btn_style,
        )

        self.scale_box = tk.Frame(self.ctrl_frame, bg=CARD_BG, padx=10, pady=4)
        self.scale_lbl = tk.Label(self.scale_box, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 10, "bold"))
        self.scale_var = tk.StringVar(value="2x")
        self.scale_combo = ttk.Combobox(
            self.scale_box,
            textvariable=self.scale_var,
            width=6,
            values=["1.5x", "2x", "3x", "4x", "6x", "8x"],
            font=("Segoe UI", 10, "bold"),
            style="Lagrange.TCombobox",
        )
        self.scale_combo.bind("<<ComboboxSelected>>", self.refresh_dim_labels)
        self.scale_combo.bind("<KeyRelease>", self.refresh_dim_labels)
        self.scale_combo.bind("<Return>", lambda _e: self.run_lagrange_interpolation())

        self.method_box = tk.Frame(self.ctrl_frame, bg=CARD_BG, padx=10, pady=4)
        self.method_lbl = tk.Label(self.method_box, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 10, "bold"))
        self.method_combo = ttk.Combobox(
            self.method_box,
            state="readonly",
            width=28,
            font=("Segoe UI", 10),
            style="Lagrange.TCombobox",
            values=TEXTS["rtl"]["methods"],
        )
        self.method_combo.current(1)

        self.run_btn = tk.Button(
            self.action_frame,
            bg=GREEN,
            fg="#11111B",
            activebackground="#94E2D5",
            command=self.run_lagrange_interpolation,
            **btn_style,
        )
        self.save_btn = tk.Button(
            self.action_frame,
            bg=YELLOW,
            fg="#11111B",
            activebackground="#FAB387",
            command=self.save_output_image,
            state="disabled",
            **btn_style,
        )

        # بطاقتا عرض الصور
        self.cards_frame = tk.Frame(self.root, bg=APP_BG, padx=20, pady=8)
        self.cards_frame.pack(fill="both", expand=True)
        self.cards_frame.columnconfigure(0, weight=1, uniform="card")
        self.cards_frame.columnconfigure(1, weight=1, uniform="card")
        self.cards_frame.rowconfigure(0, weight=1)

        self.orig_panel = self.make_card(BLUE, "orig_empty", "orig")
        self.res_panel = self.make_card(GREEN, "res_empty", "res")

    def make_card(self, accent: str, empty_key: str, which: str) -> PreviewPanel:
        card = tk.Frame(self.cards_frame, bg=CARD_BG, padx=12, pady=12)
        title = tk.Label(card, bg=CARD_BG, fg=accent, font=("Segoe UI", 12, "bold"))
        title.pack(fill="x", pady=(0, 8))

        holder = tk.Frame(card, bg=PREVIEW_BG, width=320, height=260)
        holder.pack(fill="both", expand=True)
        holder.pack_propagate(False)

        view = tk.Label(holder, bg=PREVIEW_BG, fg=MUTED, font=("Segoe UI", 10), justify="center")
        view.place(relx=0.5, rely=0.5, anchor="center")

        dim = tk.Label(card, bg=CARD_BG, fg=SUBTLE, font=("Segoe UI", 10, "bold"), pady=6)
        dim.pack(fill="x")

        view_btn = tk.Button(
            card,
            bg=CTRL_BG,
            fg=TEXT,
            activebackground=CTRL_ACTIVE,
            activeforeground="#FFFFFF",
            disabledforeground=MUTED,
            font=("Segoe UI", 9, "bold"),
            bd=0,
            padx=10,
            pady=5,
            cursor="hand2",
            state="disabled",
            command=lambda: self.open_viewer(which),
        )
        view_btn.pack(fill="x")

        panel = PreviewPanel(
            empty_key=empty_key,
            card=card,
            title=title,
            holder=holder,
            view=view,
            dim=dim,
            view_btn=view_btn,
        )
        holder.bind("<Configure>", lambda _e, p=panel: self.schedule_preview(p))
        return panel

    def bind_shortcuts(self) -> None:
        self.root.bind("<Control-KeyPress>", self._handle_ctrl_shortcut)
        self.root.bind("<F5>", lambda _e: self.run_lagrange_interpolation())
        self.root.bind("<Control-Return>", lambda _e: self.run_lagrange_interpolation())

    def _handle_ctrl_shortcut(self, event: tk.Event) -> str | None:
        """دعم Ctrl+O و Ctrl+S سواء كانت لوحة المفاتيح بالعربية أو الإنجليزية."""
        keysym = event.keysym.lower()
        is_windows = sys.platform == "win32"
        if keysym in ("o", "arabic_khah") or (is_windows and event.keycode == 79):
            self.open_image_file()
            return "break"
        if keysym in ("s", "arabic_seen") or (is_windows and event.keycode == 83):
            self.save_output_image()
            return "break"
        return None

    # ------------------------------------------------ تبديل الاتجاه واللغة

    def toggle_direction(self) -> None:
        self.is_rtl = not self.is_rtl
        self.apply_direction_layout()

    def apply_direction_layout(self) -> None:
        t = self.t()
        self.root.title(str(t["win_title"]))
        self.title_lbl.config(text=str(t["header"]))
        self.lang_btn.config(text=str(t["lang_btn"]))
        self.load_btn.config(text=str(t["load_btn"]))
        self.scale_lbl.config(text=str(t["scale_lbl"]))
        self.method_lbl.config(text=str(t["method_lbl"]))
        self.save_btn.config(text=str(t["save_btn"]))
        self.orig_panel.title.config(text=str(t["orig_title"]))
        self.res_panel.title.config(text=str(t["res_title"]))
        self.orig_panel.view_btn.config(text=str(t["view_btn"]))
        self.res_panel.view_btn.config(text=str(t["view_btn"]))
        self.refresh_run_btn()

        selected_idx = self.method_combo.current()
        self.method_combo.config(values=t["methods"])
        self.method_combo.current(1 if selected_idx < 0 else selected_idx)

        for widget in (
            self.title_lbl,
            self.lang_btn,
            self.load_btn,
            self.scale_box,
            self.method_box,
            self.run_btn,
            self.save_btn,
            self.scale_lbl,
            self.scale_combo,
            self.method_lbl,
            self.method_combo,
        ):
            widget.pack_forget()

        side, opposite = ("right", "left") if self.is_rtl else ("left", "right")
        self.title_lbl.pack(side=side)
        self.lang_btn.pack(side=opposite)

        for widget in (self.load_btn, self.scale_box, self.method_box, self.run_btn, self.save_btn):
            widget.pack(side=side, padx=6)

        for label, combo in ((self.scale_lbl, self.scale_combo), (self.method_lbl, self.method_combo)):
            label.pack(side=side, padx=(6, 4))
            combo.pack(side=side, padx=4)

        if self.is_rtl:
            self.orig_panel.card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
            self.res_panel.card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        else:
            self.orig_panel.card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            self.res_panel.card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.status_bar.config(
            anchor="e" if self.is_rtl else "w",
            justify="right" if self.is_rtl else "left",
        )
        self.render_preview(self.orig_panel)
        self.render_preview(self.res_panel)
        self.refresh_dim_labels()
        self.render_status()

    def refresh_run_btn(self) -> None:
        t = self.t()
        if self.busy:
            self.run_btn.config(text=str(t["cancel_btn"]), bg="#F38BA8", activebackground="#EBA0AC")
        else:
            self.run_btn.config(text=str(t["run_btn"]), bg=GREEN, activebackground="#94E2D5")

    def refresh_dim_labels(self, *_args: object) -> None:
        t = self.t()
        if self.orig_arr is None:
            self.orig_panel.dim.config(text="—")
            self.res_panel.dim.config(text="—")
            return

        h, w = self.orig_arr.shape[:2]
        self.orig_panel.dim.config(text=str(t["dim_badge"]).format(w=w, h=h))

        if self.res_arr is not None:
            rh, rw = self.res_arr.shape[:2]
            self.res_panel.dim.config(text=str(t["dim_badge"]).format(w=rw, h=rh))
        else:
            try:
                scale = parse_scale(self.scale_var.get())
                self.res_panel.dim.config(
                    text=str(t["dim_expected"]).format(w=round(w * scale), h=round(h * scale))
                )
            except ValueError:
                self.res_panel.dim.config(text="—")

    # ------------------------------------------------ إدارة المعاينة

    def set_preview(self, panel: PreviewPanel, arr: np.ndarray | None) -> None:
        panel.cached_thumb = None
        if arr is not None:
            thumb = Image.fromarray(arr)
            thumb.thumbnail((PREVIEW_CACHE_PX, PREVIEW_CACHE_PX), RESAMPLE_DOWN)
            panel.cached_thumb = thumb

        panel.view_btn.config(state="normal" if arr is not None else "disabled")
        self.render_preview(panel)

    def schedule_preview(self, panel: PreviewPanel) -> None:
        if panel.resize_after_id is not None:
            self.root.after_cancel(panel.resize_after_id)
        panel.resize_after_id = self.root.after(60, lambda: self.render_preview(panel))

    def render_preview(self, panel: PreviewPanel) -> None:
        panel.resize_after_id = None
        if panel.cached_thumb is None:
            panel.photo = None
            panel.view.config(image="", text=str(self.t()[panel.empty_key]))
            return

        canvas_w = max(panel.holder.winfo_width() - 8, 40)
        canvas_h = max(panel.holder.winfo_height() - 8, 40)
        img_w, img_h = panel.cached_thumb.size
        fit_ratio = min(canvas_w / img_w, canvas_h / img_h)
        target_size = (max(1, round(img_w * fit_ratio)), max(1, round(img_h * fit_ratio)))

        resample = RESAMPLE_UP if fit_ratio > 1 else RESAMPLE_DOWN
        shown = panel.cached_thumb.resize(target_size, resample)
        panel.photo = ImageTk.PhotoImage(shown)
        panel.view.config(image=panel.photo, text="")

    def open_viewer(self, which: str) -> None:
        arr = self.orig_arr if which == "orig" else self.res_arr
        if arr is None:
            return
        t = self.t()
        title_key = "orig_title" if which == "orig" else "res_title"
        title = str(t[title_key]).replace("‏", "")
        PanViewer(self.root, arr, title, str(t["viewer_hint"]))

    # ------------------------------------------------ فتح وحفظ الملفات

    def open_image_file(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title=str(self.t()["open_title"]),
            initialdir=os.path.dirname(self.src_path) or None,
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        t = self.t()
        try:
            arr, has_alpha, icc = load_image_array(path)
        except Exception as exc:
            messagebox.showerror(str(t["err_title"]), str(t["err_open"]).format(msg=exc))
            return

        self.orig_arr, self.has_alpha, self.icc = arr, has_alpha, icc
        self.src_path = path
        self.res_arr = None
        self.save_btn.config(state="disabled")
        self.progress["value"] = 0
        self.set_preview(self.orig_panel, arr)
        self.set_preview(self.res_panel, None)
        self.refresh_dim_labels()

        h, w = arr.shape[:2]
        self.set_status("status_loaded", w=w, h=h, name=os.path.basename(path))

    # ------------------------------------------------ تنفيذ التكبير في الخلفية

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.refresh_run_btn()
        self.load_btn.config(state="disabled" if busy else "normal")
        self.scale_combo.config(state="disabled" if busy else "normal")
        self.method_combo.config(state="disabled" if busy else "readonly")
        self.save_btn.config(state="disabled" if busy or self.res_arr is None else "normal")

    def run_lagrange_interpolation(self) -> None:
        t = self.t()
        if self.busy:
            self.cancel_event.set()
            return

        if self.orig_arr is None:
            messagebox.showwarning(str(t["warn_title"]), str(t["warn_no_img"]))
            return

        try:
            scale = parse_scale(self.scale_var.get())
        except ValueError:
            messagebox.showerror(str(t["err_title"]), str(t["err_scale"]))
            return

        h, w = self.orig_arr.shape[:2]
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        megapixels = new_w * new_h / 1e6

        if megapixels > MAX_MEGAPIXELS:
            messagebox.showerror(
                str(t["err_title"]),
                str(t["err_too_big"]).format(mp=megapixels, max=MAX_MEGAPIXELS),
            )
            return

        if megapixels > CONFIRM_MEGAPIXELS and not messagebox.askyesno(
            str(t["warn_title"]),
            str(t["confirm_big"]).format(w=new_w, h=new_h, mp=megapixels),
        ):
            return

        points = POINTS_CHOICES[max(0, self.method_combo.current())]
        self.scale, self.job_points = scale, points
        self.cancel_event = threading.Event()
        work_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.result_queue = work_queue
        self.start_time = time.perf_counter()
        self.progress["value"] = 0
        self.set_busy(True)
        self.set_status("status_working", p=0)

        source_image = self.orig_arr
        cancel_flag = self.cancel_event

        def worker() -> None:
            try:
                resized = lagrange_resize(
                    source_image,
                    new_w,
                    new_h,
                    points,
                    progress=lambda frac: work_queue.put(("progress", frac)),
                    cancel_event=cancel_flag,
                )
                work_queue.put(("done", resized))
            except Cancelled:
                work_queue.put(("cancelled", None))
            except MemoryError:
                work_queue.put(("error", str(t["err_memory"])))
            except Exception as exc:
                work_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self.poll_job)

    def poll_job(self) -> None:
        work_queue = self.result_queue
        if work_queue is None:
            return

        last_progress: float | None = None
        final_event: tuple[str, object] | None = None

        try:
            while True:
                kind, payload = work_queue.get_nowait()
                if kind == "progress":
                    last_progress = float(payload)
                else:
                    final_event = (kind, payload)
                    break
        except queue.Empty:
            pass

        if last_progress is not None:
            percent = int(last_progress * 100)
            self.progress["value"] = percent
            if final_event is None:
                self.set_status("status_working", p=percent)

        if final_event is None:
            self.root.after(50, self.poll_job)
            return

        self.result_queue = None
        kind, payload = final_event
        t = self.t()

        if kind == "done":
            assert isinstance(payload, np.ndarray)
            self.res_arr = payload
            self.progress["value"] = 100
            self.set_preview(self.res_panel, payload)
            self.set_busy(False)
            self.refresh_dim_labels()
            rh, rw = payload.shape[:2]
            self.set_status(
                "status_done",
                s=f"{self.scale:g}",
                n=self.job_points,
                w=rw,
                h=rh,
                sec=time.perf_counter() - self.start_time,
            )
        elif kind == "cancelled":
            self.progress["value"] = 0
            self.set_busy(False)
            self.set_status("status_cancelled")
        else:
            self.progress["value"] = 0
            self.set_busy(False)
            self.set_status("status_error", msg=payload)
            messagebox.showerror(str(t["err_title"]), str(t["err_run"]).format(msg=payload))

    def save_output_image(self) -> None:
        if self.res_arr is None or self.busy:
            return
        t = self.t()
        base_name = os.path.splitext(os.path.basename(self.src_path))[0] or "image"
        path = filedialog.asksaveasfilename(
            title=str(t["save_title"]),
            defaultextension=".png",
            initialdir=os.path.dirname(self.src_path) or None,
            initialfile=f"{base_name}_x{self.scale:g}.png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg *.jpeg"),
                ("WebP Image", "*.webp"),
                ("TIFF Image", "*.tif *.tiff"),
                ("BMP Image", "*.bmp"),
            ],
        )
        if not path:
            return
        try:
            save_image_array(self.res_arr, path, self.icc)
        except Exception as exc:
            messagebox.showerror(str(t["err_title"]), str(t["err_save"]).format(msg=exc))
            return
        self.set_status("status_saved", path=path)


# ---------------------------------------------------------------- نقطة الدخول (GUI + CLI)


def run_cli(input_path: str, output_path: str, scale_text: str, points: int) -> None:
    """تشغيل التكبير مباشرة من سطر الأوامر عند تحديد مسار الإخراج."""
    scale = parse_scale(scale_text)
    arr, _, icc = load_image_array(input_path)
    h, w = arr.shape[:2]
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))

    t0 = time.perf_counter()
    resized = lagrange_resize(arr, new_w, new_h, points=points)
    save_image_array(resized, output_path, icc)
    elapsed = time.perf_counter() - t0
    print(f"Done ({scale:g}x, {points}-point): {w}x{h} -> {new_w}x{new_h} in {elapsed:.2f}s -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="تكبير الصور باستيفاء لاغرانج | Lagrange Image Resizer"
    )
    parser.add_argument("image", nargs="?", help="مسار الصورة لفتحها في الواجهة أو معالجتها")
    parser.add_argument("-o", "--output", help="مسار الحفظ (يفعّل وضع سطر الأوامر بدون واجهة)")
    parser.add_argument("-s", "--scale", default="2", help="معامل التكبير بين 1 و 8 (الافتراضي: 2)")
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        choices=POINTS_CHOICES,
        default=4,
        help="رتبة الاستيفاء: 2 خطي، 4 تكعيبي، 6 خماسي (الافتراضي: 4)",
    )
    args = parser.parse_args()

    if args.output:
        if not args.image:
            parser.error("يجب تحديد مسار الصورة المدخلة عند استخدام -o / --output")
        run_cli(args.image, args.output, args.scale, args.points)
        return

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    app = LagrangeImageResizerApp(root)
    if args.image and os.path.isfile(args.image):
        root.after(100, lambda: app.load_path(args.image))
    root.mainloop()


if __name__ == "__main__":
    main()