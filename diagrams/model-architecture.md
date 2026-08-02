# Model architecture: two-stage inference pipeline

Why two stages: InsPLAD's detection annotations carry no condition labels
and its condition labels carry no boxes, so one model per question. Stage 1
answers "what and where," stage 2 answers "what condition."

```mermaid
flowchart TD
    IMG["UAV image<br/>(any resolution)"] --> PRE["Letterbox resize to 640 x 640<br/>pad value 114, normalize 0-1"]

    subgraph stage1["Stage 1 · Detector · YOLO11-s, INT8 ONNX, 10.1 MB"]
        PRE --> BB["Backbone + neck<br/>(CSP blocks, SPPF, C2PSA)"]
        BB --> HEAD["Detection head<br/>3 scales"]
        HEAD --> OUT1["Boxes + class + confidence<br/>17 asset classes"]
    end

    OUT1 --> NMS["NMS + confidence filter"]
    NMS --> ROUTE{"Asset type<br/>defect-prone?"}

    ROUTE -->|"12 classes: no"| FIND1["Finding:<br/>asset + location + confidence"]
    ROUTE -->|"5 classes: yes"| CROP["Crop box region<br/>resize 224, ImageNet normalize"]

    subgraph stage2["Stage 2 · Condition classifier · EfficientNetV2-S, FP32 ONNX, 82 MB"]
        CROP --> EFF["EfficientNetV2-S backbone<br/>ImageNet pretrained, fine-tuned"]
        EFF --> SOFT["Softmax over 11<br/>asset__condition classes<br/>logits masked to the detected<br/>asset's valid conditions"]
    end

    SOFT --> FIND2["Finding: asset + location<br/>+ condition + confidences"]

    FIND1 --> OUT["Findings list"]
    FIND2 --> OUT
```

Defect-prone asset types and their conditions:

| Asset | Conditions |
|---|---|
| glass insulator | good · missing-cap |
| lightning rod suspension | good · rust |
| polymer insulator upper shackle | good · rust |
| vari-grip | good · rust · bird-nest |
| yoke suspension | good · rust |

Quantization scope is detector-only: it is the model that sees the full
image every frame and dominates compute. The classifier only runs on small
crops and stays FP32.
