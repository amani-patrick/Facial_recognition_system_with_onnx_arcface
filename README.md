# Face Recognition with ArcFace ONNX and 5-Point Alignment

A modular, CPU-only face recognition system using ArcFace ONNX model with 5-point facial landmark alignment. Features clean code with descriptive naming, suitable for education and practical use on laptops/desktops.

## Setup

1. Install uv: `pip install uv` or use the official installer
2. Initialize project: `uv sync`
3. Set up structure: `uv run python setup_proj.py`

## Usage

- Enroll: `uv run python -m src.enroll`
- Recognize: `uv run python -m src.recognize`
- Test modules: `uv run python -m src.camera`, etc.

## License

Educational and non-commercial use encouraged.
