**Date:** February 9th, 2025
**Author:** Zahra

# Literature Review (Fault/Failure Types Mentioned or Injected) 

## 1) Fault Injection / Robustness Testing *in LLM-MAS*

### **AEGIS — Automated Error Generation & Attribution**
**[Kong, 2025]** introduces an automated pipeline that generates **faulty multi-agent trajectories** by taking successful traces and applying **controlled, context-aware error injections** via an LLM “manipulator,” producing a labeled dataset for error attribution.  
**Faults injected (MAST-aligned “Functional Mistakes” used as injection prompts):**
- **FM-1.x Task/Execution Errors**
  - **FM-1.1 Task specification deviation**ƒƒƒƒ
  - **FM-1.2 Role specification deviation**
  - **FM-1.3 Add redundant steps** (looping/repetition)
  - **FM-1.4 Remove conversation history** (amnesia/context wipe)
  - **FM-1.5 Remove termination conditions** (non-termination / endless output)
- **FM-2.x Communication/Coordination Errors**
  - **FM-2.1 Repeat handled tasks** (restart / redo already done work)
  - **FM-2.2 Make request ambiguous** (unsafe assumptions / missing clarification)
  - **FM-2.3 Deviate from main goal** (drift to irrelevant path)
  - **FM-2.4 Hide important information** (intentional withholding)
  - **FM-2.5 Ignore other agents** (reject feedback/corrections)
  - **FM-2.6 Inconsistent reasoning** (reasoning ≠ action contradictions)
- **FM-3.x Quality/Verification Errors**
  - **FM-3.1 Premature termination**
  - **FM-3.2 Remove verification steps**
  - **FM-3.3 Incorrect verification** (verifies wrong results as correct)

---

### **AgenTracer — Failure Attribution via Counterfactual Replay + Programmatic FI**
**[Zhang, 2025]** proposes a failure-attribution framework that labels decisive error steps using **counterfactual replay**, and expands data by **programmatic fault injection** that perturbs successful trajectories into synthetic failures.  
**Faults injected (programmatic, step-level corruption rather than a fixed taxonomy):**
- **Targeted corruption of a selected step** (injecting a failure at a chosen agent action)
  - e.g., inserting **specific incorrect code changes**, wrong return values, duplicated outputs, removed logic, etc.
- **“Attack expert” style perturbations**: injection prompt explicitly demands **concrete, implementable modifications** to flip success → failure
- Injection results in synthetic traces where the faulty agent/step is known **by construction**

---

### **Chaos Engineering for LLM-MAS**
**[Owotogbe, 2025]** argues that LLM-MAS behave like complex distributed systems and proposes adapting **chaos engineering** to test resilience under production-like disruptions.  
**Faults/disruptions proposed for chaos experiments (high level):**
- **Agent failures** (agent crash/unavailability)
- **Agent communication failures** (message loss / interaction breakdown)
- **Communication delays** (latency injected into coordination)
- **Cascading faults** (failure propagation across agents/tools)
- **Resource contention / operational disruption** (implied as part of chaos methodology)

---

## 2) Failure Root Cause Taxonomies & Failure Attribution (LLM-MAS)

### **AgentFail — Root Cause Dataset + Taxonomy + Injection Benchmark**
**[Ma, 2025]** studies failures in **platform-orchestrated agentic systems** (e.g., low-code agent platforms) and provides a taxonomy + benchmark, including experiments that measure impact by **injecting each root cause type**.  
**Taxonomy (Figure “Failure Root Cause Taxonomy”) and injected root-cause classes:**
- **Agent-level failures**
  - **F1.1 Tool/action planning error** (wrong tool choice, wrong action ordering)
  - **F1.2 Response format error** (invalid/unparsable output)
  - **F1.3 Response content deviation** (off-topic, ignores constraints, redundant output)
  - **F1.4 Knowledge/reasoning limitation** (missing knowledge, false conclusion)
  - **F1.5 Poor prompt design** (ambiguous roles, missing examples, unclear format)
  - **F1.6 Language/encoding issue** (symbols/emojis/encoding incompatibility)
  - **F1.7 Tool invocation / KB retrieval error** (internal retrieval/API call failures)
- **Workflow-level failures**
  - **F2.1 Missing input verification** (no checks on variable presence/type/format)
  - **F2.2 Unreasonable node dependency** (downstream depends on unavailable data)
  - **F2.3 Loops & deadlocks** (cyclic invocation / infinite execution)
  - **F2.4 Faulty conditional judgment** (wrong branch / misrouted path)
  - **F2.5 Improper task decomposition** (bad split of work; subtask mistakes)
  - **F2.6 Context conflict** (history/intermediate result mismatch)
  - **F2.7 Cross-agent tool/interface mismatch** (incompatible structures break parsing)
- **Platform-level failures**
  - **F3.1 Network/resource fluctuation** (latency spikes, insufficient compute)
  - **F3.2 Service unavailability** (model/API/platform runtime instability)

---

### **MAST — Why Do Multi-Agent LLM Systems Fail?**
**[Cemri, 2025]** empirically studies MAS failures and introduces a structured failure-mode taxonomy aligned with reasoning & coordination breakdowns (widely reused by later work like AEGIS).  
**Key failure types (same FM set reused by AEGIS):**
- Task/role deviation, redundant/looping actions, context/history loss, missing termination
- Coordination breakdown (repeat work, ambiguity, goal drift, hiding info, ignoring peers, inconsistency)
- Verification breakdown (premature stop, no verification, incorrect verification)

---

### **TRAIL — Trace Reasoning & Agentic Issue Localization**
**[Deshpande, 2025]** introduces a **formal taxonomy** for debugging structured agent traces and releases a dataset of **148 human-annotated traces** grounded in SWE-Bench/GAIA scenarios.  
**Taxonomy of error types (explicitly listed as a tree in the paper):**

**Reasoning Errors**
- **Hallucinations**
  - language-only hallucinations  
  - tool-related hallucinations (fabricated tool capabilities/outputs)
- **Information Processing**
  - **Poor information retrieval** (retrieves irrelevant info)
  - **Tool output misinterpretation** (wrong assumptions/context for tool output)
- **Decision Making**
  - **Incorrect problem identification** (misunderstands global/local task)
  - **Tool selection errors** (uses wrong tool)

**Output Generation Errors**
- **Formatting errors** (invalid code/JSON/structured output formatting)
- **Instruction non-compliance** (does something else than instructed)

**System Execution Errors**
- **Configuration**
  - **Tool definition issues** (tool described incorrectly / misleading definitions)
  - **Environment setup errors** (permissions, missing keys/resources)
- **API Issues**
  - **Rate limiting (429)**
  - **Authentication errors (401/403)**
  - **Service errors (500)**
  - **Resource not found (404)**
- **Resource management**
  - **Resource exhaustion** (memory overflow)
  - **Timeout issues**

**Planning & Coordination Errors**
- **Context management**
  - **Context handling failures** (window overflow, state tracking, forgetting)
  - **Resource abuse** (excessive tool calls due to memory/context issues)
- **Task management**
  - **Goal deviation**
  - **Task orchestration errors** (subtask coordination, progress monitoring)

**Domain-specific errors**
- Failures specific to the task domain

---

### **Who&When — Failure Attribution Dataset**
**[Zhang, 2025]** proposes “automated failure attribution” and releases the **Who&When dataset**, which contains failure logs from many LLM-MAS systems with annotations for **which agent** and **which step** is decisive.  
**Fault/failure types included:**  
- This paper focuses on **attribution**, not a formal taxonomy; faults are described in **natural-language reasons** + decisive agent/step labels, rather than a standardized fault type set.

---

## 3) Observability / Ops / Security-Failure Awareness in Agentic Systems

### **LumiMAS — Monitoring & Observability Framework**
**[Solomon, 2025]** proposes a real-time monitoring and observability framework for MAS and explicitly links MAS risks to OWASP-style agentic threats.  
**Failure/fault types emphasized (observability-driven):**
- **Prompt injection**
- **Memory poisoning**
- **Cascading hallucinations**
- Broader monitoring of anomalous agent behavior and unsafe coordination patterns

---

### **AgentOps — Observability Taxonomy for LLM Agents**
**[Dong, 2024]** proposes an AgentOps taxonomy (artifacts + telemetry across lifecycle) based on mapping existing tools, focusing on what needs to be traced for safety and reliability.  
**Failure/fault types explicitly mentioned or implied as monitorable risks:**
- **Code injection attacks in prompts**
- **Secret leakage embedded in prompts**
- **Unexpected results / trace-level errors**
- Guardrail-triggered events (blocks/filters/fallback escalation) indicating failures or near-failures

---

### **Survey: LLM-based Multi-Agent Systems**
**[Li, 2024]** surveys architectures, workflows, infrastructure, and challenges of LLM-MAS across profile/perception/action/interaction/evolution.  
**Failure/fault types highlighted as major challenges (useful as injection targets even if not injected in the paper):**
- **Hallucination** (intrinsic/extrinsic)
- **Bias / knowledge update issues**
- **Memory failures** (forgetting, overwrite, uncontrolled memory growth)
- **Coordination failures** (communication overhead, inconsistent agent states)
- **Tool-use failures** (wrong tool calls, incorrect grounding, tool misinterpretation)
- **Robustness & evaluation gaps** (unstable behavior, weak failure diagnosis)
- **Safety/security issues** (implicitly connected to injection/poisoning/prompt attacks in surveyed works)

---

### **Taxonomy of Failure Modes in Agentic AI Systems (Whitepaper)**
**[Bryan, n.d.]** provides an industry-oriented taxonomy of how agentic systems fail, intended as a practical reference for engineering teams.  
**Failure/fault types (high-level categories; useful injection inspiration):**
- Planning/control failures (goal drift, looping, dead-ends)
- Tool-use failures (wrong tool, unsafe invocation, tool output misuse)
- Context/memory failures (forgetting, stale state, misleading context)
- Safety failures (prompt injection, data leakage, policy violations)
- System/runtime failures (timeouts, dependency failures, resource issues)

---

## 4) Non-MAS (but highly relevant inspiration for fault injection design)

### **Service-Level Fault Injection Testing (Filibuster)**
**[Meiklejohn, 2021]** proposes “service-level fault injection testing,” injecting realistic failures into microservice RPC calls using instrumentation (inspired by observability tooling), automatically exploring failure combinations.  
**Faults injected / emphasized:**
- **Call-site failures** (client library exceptions)
  - notably **timeout** and **connection error**
- **Remote service failures / error responses**
  - injected **HTTP error responses** (4xx/5xx), e.g., 403/404/500/503
- Production-inspired failure patterns (discussed as real bugs):
  - **misconfigured timeouts**
  - **fallbacks to the same server** (bad redundancy)
- Also supports conditional fault injection (wait timeout duration then fail, etc.)

---

### **SLR: Fault Injection Testing of Microservice Systems**
**[Yu, 2025]** reviews fault injection techniques for microservices and summarizes common fault models and where the literature is lacking.  
**Basic fault types commonly injected (Table “Basic Fault Types…”):**
- **Crash**
- **Hang**
- **Overload**
- **Disconnect**
- **Latency**
**Noted as missing/underexplored (unique microservice faults; good future injection targets):**
- **Dynamic scaling faults**
- **Container image corruption**
- **Configuration drift**

---

### **Greybox Fuzzing of Distributed Systems (Mallory)**
**[Meng, 2023]** proposes a greybox fuzzing framework that explores distributed system behaviors by perturbing execution schedules and communication patterns (feedback-guided).  
**Faults/failure triggers commonly used in DS fuzzing (injection inspiration):**
- Message-level faults: **drop**, **delay**, **reorder**, **duplicate**
- Node/process faults: **crash/kill**, **restart**
- Network-level faults: **partition**, **disconnect**, **latency injection**
- Schedule perturbation to expose hidden concurrency/state bugs

---

### **Survey: Failure Analysis & Fault Injection in AI Systems**
**[Yu, 2026]** surveys fault injection and failure analysis across AI systems (not MAS-specific), providing a broad taxonomy of *what* can fail and *how* injection is done.  
**Fault categories useful to derive injectable MAS faults:**
- **Data-level faults** (noise, missing values, poisoning, distribution shift)
- **Model-level faults** (parameter perturbation, neuron/weight fault, quantization issues)
- **Software faults** (logic bugs in preprocessing/postprocessing, integration)
- **Hardware faults** (bit flips, memory errors, GPU faults)
- **Runtime/environment faults** (resource exhaustion, latency spikes, unavailable dependencies)
- **Adversarial attacks** (input perturbation, prompt-like injection analogs)