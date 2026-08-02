# App architecture: Gradio demo on Hugging Face Spaces (planned)

The human-in-the-loop story: one confidence threshold decides what the AI
auto-accepts vs what gets routed to a person. The slider makes that
business trade-off tangible.

```mermaid
flowchart TD
    subgraph browser["User's browser"]
        UP["Image upload<br/>or preloaded examples"]
        SLIDER["Confidence threshold slider<br/>0.05 - 0.95"]
        VIEW["Annotated image<br/>+ accepted findings"]
        QUEUE["Human review queue panel<br/>crop thumbnails + scores"]
        PR["Precision / recall readout<br/>at current threshold"]
        DL["GeoJSON download"]
    end

    subgraph space["Hugging Face Space · free CPU tier · Gradio"]
        subgraph inference["Inference · ONNX Runtime, runs once per image"]
            DET["detector_int8.onnx"]
            CLS["classifier.onnx + classes.json"]
            DET --> CLS
        end
        CACHE["Cached detections<br/>for current image"]
        GATE{"conf >= threshold?"}
        PRTAB["pr_table.json<br/>precomputed on test set"]
        GEO["GeoJSON builder<br/>simulated corridor coords<br/>near Tucson, AZ - labeled as such"]
    end

    UP --> DET
    CLS --> CACHE
    CACHE --> GATE
    SLIDER -->|"re-filter only,<br/>no re-inference"| GATE
    GATE -->|yes| VIEW
    GATE -->|no| QUEUE
    SLIDER --> PRTAB --> PR
    VIEW --> GEO
    QUEUE --> GEO
    GEO --> DL
```

Design decisions carried from the spec:

- Inference runs once per image; the slider re-filters cached detections
  client-side of the model, so dragging it is instant.
- The gate uses stage-1 detection confidence only. Classifier output is
  displayed but does not drive routing: one threshold, one clear story.
- The P/R readout is dataset-level (held-out test set), labeled as such in
  the UI, not a per-image number.
- No Ultralytics/torch at serve time; ONNX Runtime only, so the Space
  stays light and cold-starts reliably.
- GeoJSON: FeatureCollection of Point features with asset, condition,
  confidence, image id, timestamp; coordinates are simulated and marked
  simulated in both the file and the UI.
