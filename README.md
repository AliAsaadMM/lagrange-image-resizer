# Lagrange Image Resizer

A Python desktop application and command-line utility for image upscaling and resolution enhancement using **separable 2D Lagrange Polynomial Interpolation**. Includes a bilingual **RTL (Arabic) / LTR (English)** graphical interface, banded memory-safe processing, and a 1:1 pixel inspection viewer.

---

## Mathematical Formulation

To compute each pixel value in the target grid, its center-aligned coordinate $u$ is mapped back to the source image grid. The nearest $N$ neighboring pixels along each axis are selected as interpolation nodes to evaluate the Lagrange basis polynomials:

$$y(u) = \sum_{j=0}^{N-1} y_j \cdot L_j(u), \qquad L_j(u) = \prod_{\substack{m=0 \\ m \neq j}}^{N-1} \frac{u - x_m}{x_j - x_m}$$

Because an image is a 2D regular grid, interpolation is applied **horizontally** first and then **vertically** in row bands, significantly reducing time complexity and peak memory usage.

### Supported Interpolation Orders

| Order | Nodes ($N$) | Description |
| :--- | :---: | :--- |
| **Linear** | `2` | Fast 2-point interpolation along each axis. |
| **Cubic** | `4` | **Default** — balanced sharpness and smooth edge reconstruction. |
| **Quintic** | `6` | Higher-order polynomial fit for fine details and smooth gradients. |

---

## Features

- **Bilingual UI (RTL / LTR):** Instant switching between Arabic (RTL) and English (LTR) layouts.
- **Banded Memory-Safe Processing:** Processes images in configurable row bands (`BAND_ELEMENTS`) with megapixel guardrails and mid-run cancellation support.
- **Image Fidelity Preservation:**
  - **Premultiplied Alpha (`RGBA`):** Prevents dark edge halos when upscaling transparent PNG/WebP images.
  - **EXIF Auto-Orientation:** Automatically corrects rotated smartphone photos on load.
  - **ICC Color Profile:** Preserves embedded color profiles when exporting.
- **1:1 Detail Pan Viewer:** Inspect full-resolution output at 100% scale with mouse drag, scroll wheel, or arrow keys.
- **Flexible Scale Input:** Supports integers, decimals, `x` suffixes, and Eastern Arabic numerals (e.g., `2`, `2.5`, `2x`, `٣`).
- **Dual GUI & CLI Modes:** Launch the desktop interface or batch-resize directly from the terminal.

---

## Project Structure

```text
lagrange-image-resizer/
├── lagrange_image_resizer.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

Requires **Python 3.9+** and `tkinter` (included with standard Python installations on Windows and macOS).

```bash
git clone [https://github.com/AliAsaadMM/lagrange-image-resizer.git](https://github.com/AliAsaadMM/lagrange-image-resizer.git)
cd lagrange-image-resizer
pip install -r requirements.txt
```

---

## Usage

### Graphical Interface (GUI)

Launch the desktop app:
```bash
python lagrange_image_resizer.py
```

Or open an image directly on startup:
```bash
python lagrange_image_resizer.py path/to/image.png
```

### Command-Line Interface (CLI)

Upscale an image without opening the GUI by passing `-o` / `--output`:
```bash
python lagrange_image_resizer.py input.png -o output.png -s 3 -p 4
```

| Flag | Description | Default |
| :--- | :--- | :---: |
| `image` | Path to the input image | Required for CLI |
| `-o`, `--output` | Destination file path (enables CLI mode) | — |
| `-s`, `--scale` | Scale factor (`> 1.0` up to `8.0`) | `2` |
| `-p`, `--points` | Number of Lagrange nodes (`2`, `4`, or `6`) | `4` |

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl + O` | Open an image file |
| `Ctrl + S` | Save the resized image |
| `F5` or `Ctrl + Enter` | Start Lagrange interpolation |
| `Home` or `R` | Re-center image in the 100% Detail Viewer |
| `Esc` | Close the 100% Detail Viewer |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.