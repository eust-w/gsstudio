# GSStudio Prompt-Only Source Repair Implementation Plan

> **For implementation:** Follow the approved plan with TDD and staged rollout.

**Goal:** Make prompt-only World Generation create a real source image before panorama expansion, eliminate partially-created queued tasks, and resume matching historical failures in place.

**Architecture:** The API stages and validates uploads before inserting a pending SQLite row. A GSStudio panogen adapter loads HunyuanImage-3 once and, for prompt-only tasks, generates the source view and panorama in the same process while journaling `source_generation.json`. Startup migration requeues only the exact historical missing-input failure signature.

**Tech stack:** FastAPI, SQLite, Pillow, HunyuanImage-3/HY-Pano, React/Vitest, Kubernetes/Kaniko.

---

1. Add failing backend tests for prompt-only creation, normalized uploads, invalid input rejection, atomic visibility, FIFO scheduling, source journal, and historical requeue.
2. Implement reserved task IDs, staged upload validation, compatible DB columns, source summaries, and FIFO ordering.
3. Add a single-process panogen adapter supporting prompt-only source generation, durable hashes, resume, and isolated recovery attempts.
4. Add failing and passing frontend tests for source status, concise errors, technical details, and retry-source behavior.
5. Build and audit the application, publish immutable `a800-v42`, roll out only `gsstudio`, and verify the existing failed task plus public auth/browser behavior.
