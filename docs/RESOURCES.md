# Learning Resources

Curated links for each phase of the project. Docs first, then a deeper
article or course/video where one is worth your time. Nothing here is
required reading; grab what you need when you hit that phase.

## The domain: powerline inspection and InsPLAD

- [InsPLAD paper (arXiv 2311.01619)](https://arxiv.org/abs/2311.01619): the dataset paper. Read sections 3 and 4 to understand exactly what the annotations are and how the published baselines were measured. This is the source for every dataset claim in the README.
- [InsPLAD repository](https://github.com/andreluizbvs/InsPLAD): download links, dataset structure, license.

## Object detection and YOLO11 (detector, stage 1)

- [Ultralytics YOLO11 model docs](https://docs.ultralytics.com/models/yolo11/): what the model is and its size variants.
- [Ultralytics train mode guide](https://docs.ultralytics.com/modes/train/): every training argument used in train_detector.py, including resume, which matters on an instance that can die.
- [Roboflow: What is YOLO11](https://blog.roboflow.com/what-is-yolo11/): readable architecture overview if you want to be able to explain what changed vs earlier YOLOs in an interview.
- [Ultralytics YouTube channel](https://www.youtube.com/@Ultralytics): short practical walkthroughs of training and export.

## Dataset prep and the YOLO annotation format

- [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/): the exact YOLO txt format (class x_center y_center w h, normalized) and the dataset YAML that prep_insplad.py must produce.

## Transfer learning classifier (stage 2)

- [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): the canonical fine-tuning walkthrough; train_classifier.py is essentially this applied to InsPLAD-fault crops.
- [timm (pytorch-image-models)](https://github.com/huggingface/pytorch-image-models): pretrained backbones including EfficientNet-B0 and MobileNetV3 with a consistent API.

## Deep learning fundamentals (optional depth)

- [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html): the best from-scratch grounding available; useful if an interviewer digs below the framework layer.
- [fast.ai Practical Deep Learning](https://course.fast.ai/): top-down, code-first course; the transfer learning lessons map directly to this project.
- [Hugging Face Computer Vision course](https://huggingface.co/learn/computer-vision-course): free course covering detection, classification, and deployment.

## Cloud GPU training

- [Vast.ai docs](https://docs.vast.ai/): instance setup, SSH, and syncing checkpoints off an instance that can be reclaimed.

## ONNX export and INT8 quantization (the critical path)

- [Ultralytics export mode guide](https://docs.ultralytics.com/modes/export/): FP32 ONNX export arguments (opset, image size, NMS handling).
- [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): static vs dynamic quantization, quantize_static, and CalibrationDataReader. Read this before writing export_quantize.py; the API has shifted across versions, so also check it against the installed version.
- [ONNX Runtime Python API reference](https://onnxruntime.ai/docs/api/python/): InferenceSession usage for both the benchmark and the app.

## Evaluation metrics

- [Google ML Crash Course: classification metrics](https://developers.google.com/machine-learning/crash-course/classification): precision, recall, and thresholds; this is the vocabulary behind the app's slider.
- [Roboflow: What is mean average precision](https://blog.roboflow.com/mean-average-precision/): clear mAP50 / mAP50-95 explainer; know this cold before presenting the comparison table.

## Gradio and Hugging Face Spaces

- [Gradio quickstart](https://www.gradio.app/guides/quickstart): Blocks, events, and state, which the slider and review queue are built from.
- [Spaces overview](https://huggingface.co/docs/hub/spaces): file limits, hardware tiers, and Git LFS rules (relevant since the INT8 detector sits near the plain-file size limit).
- [Gradio Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-gradio): Space-specific config (app file, requirements, SDK version pinning).

## GeoJSON and the GIS last mile

- [RFC 7946: The GeoJSON Format](https://datatracker.ietf.org/doc/html/rfc7946): the spec; FeatureCollection, Point, and the coordinate order (longitude first) that trips everyone up once.
- [geojson.io](https://geojson.io/): paste the export here to eyeball it on a map; fastest sanity check available.
- [ArcGIS Online GeoJSON reference](https://doc.arcgis.com/en/arcgis-online/reference/geojson.htm): what ArcGIS expects on ingestion, since "shaped for ArcGIS" is a specific claim in the spec.
- [OpenStreetMap power tagging](https://wiki.openstreetmap.org/wiki/Key:power): how transmission lines are mapped in OSM; use this to find a real corridor east of Tucson for the simulated coordinates.
