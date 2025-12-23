* * *

# Face-Based Image Segregation using InsightFace

This project is a **Python script that automatically segregates images** based on whether a **specific person’s face is present or not**.

It is useful when you have:

* A large image collection
    
* Multiple photos of one known individual
    
* A need to separate images where that person appears vs doesn’t
    

The implementation uses **InsightFace**, a state-of-the-art face recognition library that handles face detection, alignment, and identity embedding in a single pipeline.

* * *

## How it Works (High Level)

1. You provide a folder containing images of the **known person**
    
2. The script builds a **reference face embedding** from those images
    
3. Each image in the dataset is scanned for faces
    
4. Any detected face is compared with the reference embedding
    
5. Images are automatically placed into:
    
    * `person_present`
        
    * `person_absent`
        

This approach is **embedding-based**, meaning it is robust to:

* Lighting changes
    
* Different angles
    
* Facial expressions
    
* Minor aging differences
    

* * *

## Why InsightFace?

* Designed specifically for face recognition (not generic object detection)
    
* Very accurate and fast
    
* Works on CPU and GPU
    
* Widely used in real-world systems
    

No YOLO, Haar cascades, or manual heuristics are used.

* * *

## Project Structure

```
.
├── segregate_faces.py
├── known_person/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── dataset/
│   ├── photo1.jpg
│   ├── photo2.png
│   └── ...
└── output/
    ├── person_present/
    └── person_absent/
```

* * *

## Installation

### Requirements

* Python 3.8 or higher
    
* Works on Windows / Linux / macOS
    

### Install dependencies

```bash
pip install insightface onnxruntime opencv-python numpy tqdm
```

For GPU support (optional):

```bash
pip install onnxruntime-gpu
```

* * *

## Running the Script

Basic usage:

```bash
python segregate_faces.py \
  --known_dir known_person \
  --dataset_dir dataset \
  --output_dir output \
  --threshold 0.45
```

### Arguments

| Argument | Description |
| --- | --- |
| `--known_dir` | Folder with images of the known person |
| `--dataset_dir` | Folder containing images to process |
| `--output_dir` | Output directory (default: `output`) |
| `--threshold` | Cosine similarity threshold (default: `0.45`) |
| `--use_gpu` | Enable CUDA if available |

Example with GPU:

```bash
python segregate_faces.py --known_dir known_person --dataset_dir dataset --use_gpu
```

* * *

## Threshold Tuning

The similarity threshold controls strictness:

| Threshold | Behavior |
| --- | --- |
| 0.35–0.40 | Very strict, fewer false positives |
| **0.45–0.55** | Balanced (recommended) |
| 0.60+ | Risk of false positives |

If images are incorrectly classified:

* **Lower the threshold** → fewer matches
    
* **Increase the threshold** → more matches
    

* * *

## Notes & Best Practices

* Use **multiple reference images** (10–20 recommended)
    
* Clear face images give better results
    
* Extremely small or blurry faces may be ignored
    
* If multiple faces exist in one image, **any match is enough** to mark it as `person_present`
    

* * *

## Logging & Progress

* Informative logs are printed to the console
    
* A progress bar shows dataset processing status
    
* Errors for unreadable images are logged and skipped safely
    

* * *

## Limitations

* Designed for **one known person**
    
* Not intended for face spoofing detection
    
* Accuracy depends on reference image quality
    

* * *

## References

* InsightFace GitHub:  
    [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
    
* InsightFace Documentation:  
    https://insightface.readthedocs.io/
    
* ArcFace Paper:  
    https://arxiv.org/abs/1801.07698
    

* * *

## License

This project is provided for educational and practical use.  
Make sure you comply with local laws and privacy regulations when processing personal images.

* * *