import os
import cv2
import shutil
import argparse
import logging
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("FaceSegregation")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def largest_face(faces):
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )


def build_reference_embedding(app : FaceAnalysis, image_dir):
    embeddings = []

    logger.info(f"Building reference embedding from: {image_dir}")

    for file in os.listdir(image_dir):
        print(f'Processing reference image: {file}')
        path = os.path.join(image_dir, file)
        img = cv2.imread(path)

        if img is None:
            logger.warning(f"Could not read image: {path}")
            continue

        faces = app.get(img)
        if not faces:
            logger.warning(f"No face found in reference image: {file}")
            continue

        face = largest_face(faces)
        embeddings.append(face.embedding)

    if not embeddings:
        raise RuntimeError("No valid reference faces found.")

    ref_embedding = np.mean(embeddings, axis=0)
    logger.info(f"Reference embedding built using {len(embeddings)} images")

    return ref_embedding


def contains_person(app, image_path, ref_embedding, threshold):
    img = cv2.imread(image_path)

    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return False

    faces = app.get(img)

    for face in faces:
        sim = cosine_similarity(face.embedding, ref_embedding)
        if sim >= threshold:
            return True

    return False


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Segregate images based on face presence")
    parser.add_argument("--known_dir", default="known", help="Directory with known person images")
    parser.add_argument("--dataset_dir", default="dataset", help="Directory with images to process")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.45, help="Cosine similarity threshold")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU (CUDA)")

    args = parser.parse_args()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.use_gpu
        else ["CPUExecutionProvider"]
    )

    logger.info("Initializing InsightFace model...")
    app = FaceAnalysis(name="buffalo_sc", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    ref_embedding = build_reference_embedding(app, args.known_dir)

    present_dir = os.path.join(args.output_dir, "person_present")
    absent_dir = os.path.join(args.output_dir, "person_absent")
    os.makedirs(present_dir, exist_ok=True)
    os.makedirs(absent_dir, exist_ok=True)

    images = [
        f for f in os.listdir(args.dataset_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    logger.info(f"Processing {len(images)} images...")
    logger.info(f"Similarity threshold: {args.threshold}")

    for img_name in tqdm(images, desc="Processing images"):
        src = os.path.join(args.dataset_dir, img_name)

        try:
            if contains_person(app, src, ref_embedding, args.threshold):
                dst = os.path.join(present_dir, img_name)
            else:
                dst = os.path.join(absent_dir, img_name)

            shutil.copy2(src, dst)

        except Exception as e:
            logger.error(f"Failed processing {img_name}: {e}")

    logger.info("Segregation completed successfully.")


if __name__ == "__main__":
    main()
