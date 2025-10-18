# ML Track

## 🧩 What You’ll Build

**Aldar Köse Storyboard Generator** — an intelligent system that **automatically generates a storyboard**: a sequence of frames telling a mini‑story about the legendary Kazakh folk hero **Aldar Köse**.

- **Input:** a short script/logline (**2–4 sentences**).
- **Output:** a **6–10 frame storyboard** that conveys a coherent story in sequence.
- **Character:** Aldar Köse must be **recognizable** and maintain a **consistent appearance** across all scenes.

You may use **any technology**, including but not limited to: - Diffusion Models, GANs - LLMs & RAG pipelines - Classical Computer Vision / Tracking / Face-ID / Style Control - Prompt‑programming, ControlNet, IP‑Adapter, style‑locking, textual inversion, DreamBooth, etc.

A short **presentation (5–8 slides)** describing your approach, pipeline, and results is required.

## 🧩 Resources Provided

- **OpenAI API key per team** (use responsibly; track usage).
- **Start in Google Colab** (recommended baseline).
- If **GPU memory is insufficient**, we will provide cloud compute credits/access on request — ping **@bayev_alen** or **@adilkhan_s** on Telegram.

## 🧩 Deliverables

1. **Inference code** (Colab + repo) that reproduces your results end‑to‑end from the given input.
2. **Model weights** if you train/customize models (or clear instructions + checkpoints path).
3. **Storyboard outputs** (PNG/JPG per frame) + an **index.json** (or CSV) with frame order and captions.
4. **5–8 slide deck** explaining approach, pipeline, experiments, and results.
5. **Reproducibility:** All artifacts must be **uploadable to S3** and reproducible from your README.

**Submission packaging:** - Upload your inference results and model weights (if exists) to **google drive** (make it publicly accessible) and include urls in your README. - Push your code to a **public GitHub repo**.

## 🧩 Judging Criteria

- **Technical implementation & ML justification — 70%**
    
    (Custom algorithms, trained or fine‑tuned models, style control, data curation, evaluation. Heavier credit for customized diffusion/GAN approaches, but strong alternative pipelines are welcome.)
    
- **Character consistency & story coherence — 20%**
    
    (Recognizable Aldar Köse; identity/style preserved across scenes; logical scene progression.)
    
- **Story quality & reproducibility — 10%**
    
    (Clarity, cultural respect, narrative flow; one‑click or scripted reproducibility.)
    

## 🧩 Recommended System Features (Inspiration)

- **Character Identity Module**: textual inversion / LoRA / DreamBooth; face/appearance anchors; CLIP‑based similarity checks.
- **Storyboard Engine**: shot list generator (LLM), scene prompts, camera/layout control (depth/pose/seg maps via ControlNet), frame‑to‑frame consistency (reference image adapters, seed locking, or latent recycling).
- **Evaluator**: automated metrics (CLIP similarity to identity refs, aesthetic score), plus human raters.
- **Repro Runner**: single command/Colab cell to regenerate from the same logline.

## 🧩 Example Directions You Can Try

1. **Style‑Locked Storyboards**: Train a tiny LoRA for Aldar Köse; generate 6–10 shots with consistent costume & face.
2. **Ref‑Guided Consistency**: Use IP‑Adapter/ControlNet to propagate identity from the first frame across subsequent frames.
3. **Data‑Lite DreamBooth**: Build a micro‑dataset (5–15 images) of Aldar Köse iconography; fine‑tune, then regularize for stability.
4. **LLM‑Driven Shotlist**: LLM produces shots + camera directions; diffusion renders; a post‑checker flags off‑style frames.
5. **Classical CV Assist**: Face tracking/feature matching to enforce continuity; reject/regenerate outliers.

## 🧩 Timeline & Format

- **Duration:** 24 hours
- **Team size:** 2–4 people
- **Kickoff → Build → Submit → Live Demos**

## 🧩 Submission Checklist

- Public **GitHub repository** with README and Colab notebook
- **Reproducible inference** (one command / one Colab cell)
- **6–10 frames** with ordered filenames and captions
- **Google Drive upload**: results + (if available) weights/checkpoints
- **5–8 slides** explaining approach & results
- Submit your results into the following url: [https://forms.gle/Tf7bKyUz3d2XidH67](https://forms.gle/HVidtzgRM9aKkMvr5)

## 🧩 Practical Notes

- Track **API usage & costs**; cache or mock when iterating.
- Prefer deterministic seeds where possible; store prompts, seeds, and control maps.
- If you use providers, document exactly **which calls** are essential vs. mockable.
- Be mindful of **cultural respect** for Aldar Köse’s image and Kazakh heritage.

## 🧩 Getting Help

- Questions: Hackathon chat / Telegram **@bayev_alen**, **@adilkhan_s**
- Compute requests: DM the contacts above.

**Good luck—and have fun building!**