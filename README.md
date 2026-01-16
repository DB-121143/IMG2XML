# Image to DrawIO (XML) Converter

This project implements a sophisticated pipeline to convert images (like flowcharts, diagrams, and technical drawings) into editable DrawIO (mxGraph) XML files. It leverages advanced Computer Vision models (**SAM 3**) and Large Language Models (**Qwen/Mistral**) to achieve high-fidelity reconstruction.

## Key Features

*   **Advanced Segmentation**: Uses **SAM 3 (Segment Anything Model 3)** for state-of-the-art segmentation of diagram elements (shapes, arrows, icons).
*   **Iterative VLM Scanning**: A structured, iterative extraction process guided by **Multimodal LLMs** ensuring no element is left behind:
    1.  **Initial Generic Extraction**: Captures standard shapes and icons.
    2.  **Refinement Rounds**: VLM scans blank areas to suggest new prompts for missed objects.
*   **High-Quality OCR (Hybrid Mode)**:
    *   **Azure Document Intelligence** for precise text localization (Bounding Boxes).
    *   **VLM (Mistral/Qwen)** for recognition correction and **LaTeX** formula conversion.
    *   **Hint Mechanism**: Uses Azure detected text as "hints" to constrain VLM generation, eliminating hallucinations and duplicate text.
    *   **Rate Limit Handling**: Smart batching (5 crops/req) with exponential backoff to handle API limits.
*   **Smart Background Removal**: Integrated **RMBG-2.0** model to automatically remove backgrounds from icons, pictures, and arrows.
*   **Arrow Handling**: Arrows are extracted as transparent images with smart masking to preserve complex routing/dashing fidelity.
*   **Vector Shape Recovery**: 
    *   **Supported Shapes**: Rectangle, Rounded Rectangle, Diamond (Decision), Ellipse (Start/End), Cylinder (Database), Cloud, Hexagon, Triangle, Parallelogram, Actor, Title Bar, Section Panel.
    *   **Color Extraction**: Intelligent "Fill vs Stroke" color extraction using statistical analysis of ROIs.
*   **Multi-User Concurrency**: 
    *   **Global Lock**: Ensures thread-safe access to GPU-heavy models.
    *   **LRU Cache**: Persists SAM3 image embeddings to allow fast interactive editing without re-encoding.

## Documentation

*   [**Technical Report**](TECHNICAL_REPORT.md): Detailed explanation of architecture, algorithms, and optimization strategies.
*   [**API Documentation**](docs/api.md) (Internal): Server API endpoints.

## Installation

### Prerequisites
*   Python 3.10+
*   Node.js & npm (for frontend)
*   CUDA-capable GPU (Recommended for SAM3/RMBG)

### Setup

1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Setup**:
    *   **RMBG-2.0**: Download `model.onnx` from [HuggingFace](https://huggingface.co/briaai/RMBG-2.0) and place in `models/rmbg/`.
    *   **SAM 3**: Update `config/config.yaml` with the path to your SAM3 checkpoint.

3.  **Environment Configuration**:
    Configure `flowchart_text/.env`:
    ```env
    # Azure Configuration (Detection)
    AZURE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
    AZURE_API_KEY=your_key

    # VLM Configuration (Recognition)
    MISTRAL_API_KEY=your_key (or compatible OpenAI key)
    MISTRAL_MODEL=qwen-vl-max (or similar)
    MISTRAL_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
    ```

## Usage

### Web Interface (Recommended)
1.  Start the backend:
    ```bash
    python server.py
    ```
2.  Start the frontend:
    ```bash
    cd frontend && npm run dev
    ```
3.  Open browser and upload images.

### Command Line
```bash
python scripts/run_all.py input/test.jpg
```
