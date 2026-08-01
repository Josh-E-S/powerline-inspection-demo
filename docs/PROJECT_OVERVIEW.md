# Powerline Inspection Demo: Plain-English Overview

This is the companion doc to the spec. The spec says *how* we build it. This explains *what* we're building, *why*, and what each piece proves.

---

## The real-world problem

Electric utilities have to inspect thousands of miles of transmission lines. Today they fly drones (or helicopters) along the lines and capture huge volumes of photos of towers, insulators, dampers, and other hardware. Then someone has to look at every photo and answer two questions:

1. **What components are in this image?** (insulator, spacer, damper, plate, etc.)
2. **What condition are they in?** (normal, corroded, broken, bird's nest on it)

Doing that manually is slow, expensive, and error-prone. A missed corroded insulator can eventually mean an outage or a wildfire. But a system that flags everything "just in case" buries human reviewers in false alarms, and they start ignoring it. That tension, missed defects vs. alarm fatigue, is the core business problem in this space.

## What we're building

A small but complete, working replica of the AI pipeline a company in this space would deploy:

```
Drone photo → AI finds components → AI classifies their condition
           → Confident findings are auto-accepted
           → Uncertain findings go to a human review queue
           → Everything exports as GeoJSON that drops into the utility's mapping (GIS) software
```

Concretely, the deliverables are:

1. **A trained detector.** We fine-tune YOLO11 (a fast, widely used object detection model) on InsPLAD, a public dataset of 10,607 real drone photos of power line hardware.
2. **A condition classifier.** For the component types that can have defects, a second small model looks at the cropped component and says "normal" or "corroded" / "broken" / "bird's nest."
3. **An optimized version of the detector.** We convert both models to ONNX and quantize the detector (the heavy one) to INT8. In plain terms: we shrink the model so it runs fast on a cheap CPU instead of needing a GPU. Then we **measure** exactly what that cost us: file size, speed, and accuracy, before vs. after.
4. **A live web demo** (Gradio app on Hugging Face Spaces, free tier) where anyone can:
   - Upload a drone image and see the detections drawn on it
   - Drag a confidence threshold slider and watch precision/recall trade off in real time
   - See low-confidence detections routed to a visible "human review queue" instead of being silently thrown away
   - Click "Export findings" and get a GeoJSON file pinned to a realistic transmission corridor near Tucson, AZ (coordinates simulated, and labeled as such)
5. **A README** with the architecture diagram, the measured comparison table, per-class error analysis, and honest framing of what this is and isn't.

## Why the threshold slider and review queue are the heart of it

Every detection the model makes comes with a confidence score. Somebody has to decide the cutoff: above it, trust the AI; below it, ask a human. That one number is a business decision, not just a technical one:

- **Set it high:** fewer false alarms, but you miss real defects. In this industry, a miss can be an outage or a fire.
- **Set it low:** you catch more, but you flood reviewers with junk and they burn out or tune it out.

The demo makes this trade-off *tangible*. You drag the slider and watch precision and recall move, and you watch detections shift between "auto-accepted" and "routed to human review." That's the conversation a Solutions Engineer has with a utility customer, turned into something you can touch.

## How each piece demonstrates production-level thinking

This is a demo-scale replica, not a production system, and it says so plainly. But each design choice mirrors what production deployment actually requires:

| Piece | What it proves |
|---|---|
| Fine-tuning on in-domain data (InsPLAD) | Real deployments start from a pretrained model and adapt it to the customer's domain. |
| Proper train/val/test splits, test set touched only once | Reported numbers are honest. Every metric survives a skeptic asking "how was this measured?" |
| ONNX INT8 quantization with in-domain calibration | Production means running within a cost budget. Shrinking a model for CPU inference, and *measuring* the accuracy cost, is core deployment work, not training work. |
| Measured benchmark table (latency p50/p95, mAP, file size) | Numbers are measured on named hardware, not estimated or copied from marketing pages. |
| Confidence gating + human review queue | No serious operator lets a model act unsupervised. Human-in-the-loop design is how AI ships in high-stakes industries. |
| GeoJSON export shaped for ArcGIS | Model output is worthless until it lands in the tool the customer already uses. Utilities live in GIS software. This is the integration "last mile." |
| Deployed on a free CPU tier, cold-start reliable | Anyone can open a URL and use it. It runs within real resource constraints, which is what the quantization bought us. |
| Honest scope section in the README | Knowing and stating a system's limits is itself a production skill, and it makes every other claim more credible. |

## What it is not (said up front, on purpose)

- It's not trained at production scale, and it doesn't claim production accuracy. Published research baselines are cited in the README so the results have context.
- Condition labels are image-level ("this insulator is corroded"), not pixel-level defect outlines. The framing is "detect components and classify condition," never "localize defects."
- The map coordinates are simulated along a real corridor for demonstration. The app and README both say so.

## The weekend plan, in one breath

Saturday: get the dataset, convert it, train on a rented GPU (~$0.25/hr, total spend under $5), then export, quantize, and benchmark. Sunday: build the Gradio app, deploy to Hugging Face Spaces, write the README with the diagram and error analysis.

## The one-line pitch

**A working end-to-end replica of an AI powerline inspection pipeline: drone imagery in, human-reviewed findings out the other side as GIS-ready data, with every number measured and every trade-off made visible.**
