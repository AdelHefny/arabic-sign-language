# Graph Report - Grand project  (2026-07-11)

## Corpus Check
- 33 files · ~14,639 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 345 nodes · 388 edges · 40 communities (24 shown, 16 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7756c843`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- app.py
- inference.py
- manifest.json
- package.json
- Arabic Sign Language Translation System — Project Summary
- package.json
- content.js
- dependencies
- devDependencies
- devDependencies
- tsconfig.json
- tsconfig.json
- offscreen.tsx
- content.tsx
- Extension Offscreen Script
- Arabic Sign Language Real-Time Translation System
- inspect_model.js
- MotionDetector
- init_offscreen.ts
- process_sentence.ts
- Docker Compose Configuration
- nlp.py
- Extension Settings UI
- main_simple.py
- Arabic Sign Language Requirements
- Backend Requirements
- MediaPipe HandLandmarker API
- Extension Logo
- Plasmo Extension README
- Project Context Summary
- Temporal Transformer Pipeline
- README.md
- graphify.md
- graphify.md

## God Nodes (most connected - your core abstractions)
1. `SentenceBuffer` - 14 edges
2. `GestureRouter` - 11 edges
3. `Arabic Sign Language Real-Time Translation System` - 11 edges
4. `DynamicInferenceEngine` - 9 edges
5. `ParticipantSession` - 9 edges
6. `Arabic Sign Language Translation System — Project Summary` - 9 edges
7. `correct_sentence()` - 8 edges
8. `_get_router()` - 7 edges
9. `StaticInferenceEngine` - 7 edges
10. `LandmarkFrame` - 6 edges

## Surprising Connections (you probably didn't know these)
- `LandmarkFrame` --uses--> `GestureRouter`  [INFERRED]
  backend/app.py → backend/inference.py
- `LandmarkFrame` --uses--> `SentenceBuffer`  [INFERRED]
  backend/app.py → backend/nlp.py
- `LandmarkSequence` --uses--> `GestureRouter`  [INFERRED]
  backend/app.py → backend/inference.py
- `LandmarkSequence` --uses--> `SentenceBuffer`  [INFERRED]
  backend/app.py → backend/nlp.py
- `ResetRequest` --uses--> `GestureRouter`  [INFERRED]
  backend/app.py → backend/inference.py

## Import Cycles
- None detected.

## Communities (40 total, 16 thin omitted)

### Community 0 - "app.py"
Cohesion: 0.09
Nodes (31): _get_buffer(), _get_router(), get_sentences(), health(), LandmarkFrame, LandmarkSequence, metrics(), predict_frame() (+23 more)

### Community 1 - "inference.py"
Cohesion: 0.09
Nodes (17): _build_session(), DynamicInferenceEngine, _load_labels(), backend/inference.py ==================== ONNX-based inference engine for both s, Recognises continuous / dynamic gestures using a temporal transformer.      Main, Add one frame of landmarks to the sliding window., Run inference on the current window.         Returns ("", 0.0) if the window is, Parameters         ----------         landmarks : 1-D float32 array of hand land (+9 more)

### Community 2 - "manifest.json"
Cohesion: 0.09
Nodes (22): action, default_icon, default_popup, background, service_worker, content_scripts, 128, 16 (+14 more)

### Community 3 - "package.json"
Cohesion: 0.09
Nodes (21): author, contributors, dependencies, plasmo, react, react-dom, description, displayName (+13 more)

### Community 4 - "Arabic Sign Language Translation System — Project Summary"
Cohesion: 0.11
Nodes (16): Arabic Sign Language Translation System — Project Summary, Architecture, Classification Metrics, Data Augmentation, Deployment Topology, Dynamic Transformer (`arabic_sign_video_model_large.onnx`), Evaluation, Goal (+8 more)

### Community 5 - "package.json"
Cohesion: 0.11
Nodes (17): author, extension_pages, description, displayName, https://meet.google.com/*, manifest, content_security_policy, host_permissions (+9 more)

### Community 6 - "content.js"
Cohesion: 0.18
Nodes (9): attachToVideo(), DEFAULTS, init(), isGoogleMeetVideo(), loadMediaPipeHands(), ParticipantSession, sessions, SETTINGS (+1 more)

### Community 7 - "dependencies"
Cohesion: 0.13
Nodes (15): @mediapipe/tasks-vision, dependencies, @mediapipe/tasks-vision, onnxruntime-node, onnxruntime-web, plasmo, @plasmohq/messaging, react (+7 more)

### Community 8 - "devDependencies"
Cohesion: 0.13
Nodes (15): devDependencies, @ianvs/prettier-plugin-sort-imports, prettier, @types/chrome, @types/node, @types/react, @types/react-dom, typescript (+7 more)

### Community 9 - "devDependencies"
Cohesion: 0.13
Nodes (15): devDependencies, @ianvs/prettier-plugin-sort-imports, prettier, @types/chrome, @types/node, @types/react, @types/react-dom, typescript (+7 more)

### Community 10 - "tsconfig.json"
Cohesion: 0.14
Nodes (13): compilerOptions, baseUrl, paths, exclude, extends, include, node_modules, .plasmo/index.d.ts (+5 more)

### Community 11 - "tsconfig.json"
Cohesion: 0.14
Nodes (13): compilerOptions, baseUrl, paths, exclude, extends, include, node_modules, .plasmo/index.d.ts (+5 more)

### Community 12 - "offscreen.tsx"
Cohesion: 0.18
Nodes (10): emptyFrameCounts, extractHands(), frameBuffers, labels, lastBufferUpdate, lastPredictions, normalizeLandmarks(), runInference() (+2 more)

### Community 13 - "content.tsx"
Cohesion: 0.27
Nodes (6): AppStatus, config, drawLandmarks(), MeetASLTranslator(), VideoOverlay(), observeParticipants()

### Community 15 - "Arabic Sign Language Real-Time Translation System"
Cohesion: 0.11
Nodes (18): 1. Install dependencies, 2. Run the backend, 3. Install the Chrome Extension, API Reference, Arabic Sign Language Real-Time Translation System, Configuration, Dynamic Recognition, Evaluation Metrics (+10 more)

### Community 23 - "nlp.py"
Cohesion: 0.18
Nodes (12): backend/config.py ================= Central configuration for the Arabic Sign La, _add_punctuation(), _apply_token_corrections(), _arabert_correct(), correct_sentence(), _load_arabert(), _normalise_whitespace(), backend/nlp.py ============== Arabic NLP post-processing pipeline.  Responsibili (+4 more)

### Community 25 - "main_simple.py"
Cohesion: 0.40
Nodes (3): predict(), PredictionRequest, BaseModel

### Community 35 - "README.md"
Cohesion: 0.50
Nodes (3): Getting Started, Making production build, Submit to the webstores

## Knowledge Gaps
- **134 isolated node(s):** `DEFAULTS`, `SETTINGS`, `sessions`, `manifest_version`, `name` (+129 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SentenceBuffer` connect `app.py` to `nlp.py`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `GestureRouter` connect `app.py` to `inference.py`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `dependencies` connect `dependencies` to `package.json`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `SentenceBuffer` (e.g. with `LandmarkFrame` and `LandmarkSequence`) actually correct?**
  _`SentenceBuffer` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `GestureRouter` (e.g. with `LandmarkFrame` and `LandmarkSequence`) actually correct?**
  _`GestureRouter` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `backend/app.py ============== Main FastAPI application.  Endpoints --------- RES`, `A single frame of hand landmarks.`, `A pre-built temporal sequence (T × D).` to the rest of the system?**
  _161 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09246088193456614 - nodes in this community are weakly interconnected._