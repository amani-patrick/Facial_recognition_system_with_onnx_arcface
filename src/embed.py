# src/embed.py
"""
Embedding stage (ArcFace ONNX) using your working pipeline:
camera
-> Haar detection
-> FaceMesh 5pt
-> align_face_5pt (112x112)
-> ArcFace embedding
-> vector visualization (education)
Run:
python -m src.embed
Keys:
q : quit
p : print embedding stats to terminal
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import time
import cv2
import numpy as np
import onnxruntime as ort
from .haar_5pt import Haar5ptDetector, align_face_5pt
# -------------------------
# Data
# -------------------------

@dataclass
class EmbeddingResult:
    embedding_vector: np.ndarray # (D,) float32, L2-normalized
    norm_before_normalization: float
    dimension: int
# -------------------------
# Embedder
# -------------------------
class ArcFaceEmbedderONNX:
    """
    ArcFace / InsightFace-style ONNX embedder.
    Input: aligned 112x112 BGR image.
    Output: L2-normalized embedding vector.
    """
    
    def __init__(
    self,
    model_path: str = "models/embedder_arcface.onnx",
    input_size: Tuple[int, int] = (112, 112),
    debug: bool = False,
):
        self.input_width, self.input_height = input_size
        self.debug = debug
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        if debug:
            print("[embed] model loaded")
            print("[embed] input:", self.session.get_inputs()[0].shape)
            print("[embed] output:", self.session.get_outputs()[0].shape)
                        
    def _preprocess(self, aligned_image: np.ndarray) -> np.ndarray:
        if aligned_image.shape[:2] != (self.input_height, self.input_width):
            aligned_image = cv2.resize(aligned_image, (self.input_width, self.input_height))
        rgb_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_image = (rgb_image - 127.5) / 128.0
        preprocessed_input = np.transpose(rgb_image, (2, 0, 1))[None, ...]
        return preprocessed_input.astype(np.float32)
    
    @staticmethod
    def _l2_normalize(vector: np.ndarray, eps: float = 1e-12):
        norm_value = float(np.linalg.norm(vector) + eps)
        return (vector / norm_value).astype(np.float32), norm_value
    
    def embed(self, aligned_image: np.ndarray) -> EmbeddingResult:
        input_tensor = self._preprocess(aligned_image)
        output_tensor = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        embedding_vector = output_tensor.reshape(-1).astype(np.float32)
        normalized_embedding, original_norm = self._l2_normalize(embedding_vector)
        return EmbeddingResult(normalized_embedding, original_norm, normalized_embedding.size)
    
# -------------------------
# Visualization helpers
# -------------------------

def draw_text_block(image, text_lines, position=(10, 30), font_scale=0.7, text_color=(0, 255, 0)):
    pos_x, pos_y = position
    for line in text_lines:
        cv2.putText(image, line, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        pos_y += int(28 * font_scale)

def draw_embedding_matrix(
    image: np.ndarray,
    embedding: np.ndarray,
    position=(10, 220),
    cell_size: int = 6,
    matrix_title: str = "embedding"
):
            
    """
    Visualize embedding vector as a heatmap matrix.
    """
    dimension = embedding.size
    columns = int(np.ceil(np.sqrt(dimension)))
    rows = int(np.ceil(dimension / columns))
    matrix = np.zeros((rows, columns), dtype=np.float32)
    matrix.flat[:dimension] = embedding
    normalized_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-6)
    grayscale_image = (normalized_matrix * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_JET)
    heatmap = cv2.resize(
        heatmap,
        (columns * cell_size, rows * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )
    
    pos_x, pos_y = position
    height, width = heatmap.shape[:2]
    image_height, image_width = image.shape[:2]
    if pos_x + width > image_width or pos_y + height > image_height:
        return 0, 0
    
    image[pos_y:pos_y+height, pos_x:pos_x+width] = heatmap
    cv2.putText(
        image,
        matrix_title,
        (pos_x, pos_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )
    return width, height


def emb_preview_str(embedding: np.ndarray, num_elements: int = 8) -> str:
    values = " ".join(f"{v:+.3f}" for v in embedding[:num_elements])
    return f"vec[0:{num_elements}]: {values} ..."

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(np.dot(vector_a, vector_b))

# -------------------------
# Demo
# -------------------------
def main():
    camera_capture = cv2.VideoCapture(0)
    
    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )
    embedding_model = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        debug=False,
    )
    
    previous_embedding: Optional[np.ndarray] = None
    print("Embedding Demo running. Press 'q' to quit, 'p' to print embedding.")
    start_time = time.time()
    frame_count = 0
    frames_per_second = 0.0
    
    while True:
        success, current_frame = camera_capture.read()
        if not success:
            break

        visualization_frame = current_frame.copy()
        detected_faces = detector.detect(current_frame, max_faces=1)
        info_lines = []

        if detected_faces:
            face = detected_faces[0]

            # draw detection
            cv2.rectangle(
                visualization_frame,
                (face.x1, face.y1),
                (face.x2, face.y2),
                (0, 255, 0),
                2,
            )

            for (x, y) in face.kps.astype(int):
                cv2.circle(visualization_frame, (x, y), 3, (0, 255, 0), -1)

            # align + embed
            aligned_face, _ = align_face_5pt(current_frame, face.kps, out_size=(112, 112))
            embedding_result = embedding_model.embed(aligned_face)

            info_lines.append(f"embedding dim: {embedding_result.dimension}")
            info_lines.append(f"norm(before L2): {embedding_result.norm_before_normalization:.2f}")

            if previous_embedding is not None:
                similarity_score = cosine_similarity(previous_embedding, embedding_result.embedding_vector)
                info_lines.append(f"cos(prev,this): {similarity_score:.3f}")

            previous_embedding = embedding_result.embedding_vector

            # aligned preview (top-right)
            aligned_thumbnail = cv2.resize(aligned_face, (160, 160))
            height, width = visualization_frame.shape[:2]
            visualization_frame[10:170, width - 170:width - 10] = aligned_thumbnail

            # --------- VISUALIZATION ---------
            draw_text_block(visualization_frame, info_lines, position=(10, 30))

            heatmap_width, heatmap_height = draw_embedding_matrix(
                visualization_frame,
                embedding_result.embedding_vector,
                position=(10, 220),
                cell_size=6,
                matrix_title="embedding heatmap",
            )

            if heatmap_width > 0:
                cv2.putText(
                    visualization_frame,
                    emb_preview_str(embedding_result.embedding_vector),
                    (10, 220 + heatmap_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )
        else:
            draw_text_block(
                visualization_frame,
                ["no face"],
                position=(10, 30),
                text_color=(0, 0, 255),
            )

        # FPS
        frame_count += 1
        delta_time = time.time() - start_time
        if delta_time >= 1.0:
            frames_per_second = frame_count / delta_time
            frame_count = 0
            start_time = time.time()

        cv2.putText(
            visualization_frame,
            f"fps: {frames_per_second:.1f}",
            (10, visualization_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Face Embedding", visualization_frame)

        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord("q"):
            break
        elif pressed_key == ord("p") and previous_embedding is not None:
            print("[embedding]")
            print(" dim:", previous_embedding.size)
            print(" min/max:", previous_embedding.min(), previous_embedding.max())
            print(" first10:", previous_embedding[:10])

    camera_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()









