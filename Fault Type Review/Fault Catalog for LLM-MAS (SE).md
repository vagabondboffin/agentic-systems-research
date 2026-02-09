**Date:** February 9th, 2025
**Author:** Zahra

---

## Legend
- **Surface**
  - **A2A**: agent↔agent
  - **A2T**: agent↔tool/API/env
  - **A2M**: agent↔memory/context/KB
  - **A2O**: agent↔orchestrator/workflow/router
  - **INFRA**: runtime/platform/network/resources
  - **OBS**: observability/telemetry pipeline
  - **MODEL**: model-level internals (weights/activations/hardware faults)
- **Type**
  - **Semantic** = meaning/decision correctness
  - **Non-semantic** = timing, availability, transport, formatting, metadata, etc.

---

# 1) Instruction / Role / Goal Faults (Semantic)

- **Disobey task specification** (A2O, Semantic) [Cemri, 2025; Kong, 2025]  
- **Disobey role specification** (A2O, Semantic) [Cemri, 2025; Kong, 2025]  
- **Instruction non-compliance** (A2O, Semantic) [Deshpande, 2025]  
- **Misinterpretation of instructions** (A2O, Semantic) [Bryan, 2025]  
- **Incorrect problem identification (wrong task understanding)** (A2O, Semantic) [Deshpande, 2025; Bryan, 2025]  
- **Goal deviation / goal drift** (A2O, Semantic) [Deshpande, 2025; Dong, 2024]  
- **Poor prompt design** (A2O, Semantic) [Ma, 2025; Dong, 2024]  
- **Prompt drift/regression over lifecycle** (A2O, Semantic-over-time) [Dong, 2024]  

---

# 2) Inter-Agent Coordination / Communication Faults

## 2.1 Semantic coordination faults
- **Fail to ask for clarification** (A2A, Semantic) [Cemri, 2025]  
- **Task derailment** (A2A/A2O, Semantic) [Cemri, 2025]  
- **Information withholding** (A2A, Semantic) [Cemri, 2025]  
- **Ignored other agent’s input** (A2A, Semantic) [Cemri, 2025]  
- **Reasoning–action mismatch** (A2A/A2T, Semantic) [Cemri, 2025]  
- **Improper task decomposition** (A2O/A2A, Semantic) [Ma, 2025]  
- **Faulty conditional judgment / wrong branch** (A2O, Semantic) [Ma, 2025]  

## 2.2 Non-semantic coordination faults
- **Conversation reset** (A2M/A2O, Non-semantic) [Cemri, 2025]  
- **Step repetition / redundant steps** (A2O, Non-semantic control-flow) [Cemri, 2025; Kong, 2025]  
- **Workflow loops and deadlocks** (A2O, Non-semantic) [Ma, 2025; Bryan, 2025]  
- **Unreasonable node dependency** (A2O, Non-semantic) [Ma, 2025]  
- **Cross-agent tool/interface mismatch** (A2A/A2T, Non-semantic) [Ma, 2025]  
- **Task orchestration errors** (A2O, Semantic or Non-semantic) [Deshpande, 2025; Ma, 2025]  

---

# 3) Memory / Context / Knowledge Base Faults

## 3.1 Non-semantic state faults
- **Loss of conversation history / truncation** (A2M, Non-semantic) [Cemri, 2025]  
- **Context handling failures (window overflow, state tracking errors)** (A2M, Non-semantic) [Deshpande, 2025]  
- **Context conflict (contradictory intermediate states passed downstream)** (A2M/A2O, Semantic outcome) [Ma, 2025]  
- **Context staleness / outdated memory used** (A2M, Non-semantic) [Li, 2024; Bryan, 2025]  
- **Memory capacity exhaustion / poor eviction strategy** (A2M, Non-semantic) [Li, 2024]  
- **Memory modification mistakes (merge/overwrite wrong info)** (A2M, Semantic) [Li, 2024]  
- **Loss of data provenance / cannot trace source behind actions** (A2M/OBS, Non-semantic → high risk) [Bryan, 2025]  

## 3.2 Adversarial memory faults
- **Memory poisoning** (A2M, Semantic security) [Solomon, 2025; Bryan, 2025]  
- **Targeted KB/RAG poisoning** (A2M, Semantic security) [Bryan, 2025; Li, 2024]  
- **Memory theft / exfiltration of stored sensitive data** (A2M, Non-semantic security) [Bryan, 2025]  

---

# 4) Reasoning / Decision Faults (Semantic)

- **Hallucinations (language-only)** (MODEL/A2O, Semantic) [Deshpande, 2025; Li, 2024; Owotogbe, 2025]  
- **Tool-related hallucinations (fabricated tool outputs/capabilities)** (A2T, Semantic) [Deshpande, 2025]  
- **Knowledge update failure / stale knowledge** (A2M, Semantic) [Li, 2024]  
- **Bias amplification / biased output** (MODEL/A2O, Semantic) [Li, 2024; Bryan, 2025]  
- **Poor information retrieval (irrelevant/overload retrieval)** (A2T, Semantic) [Deshpande, 2025]  
- **Tool output misinterpretation** (A2T, Semantic) [Deshpande, 2025]  
- **Tool selection error** (A2T, Semantic) [Deshpande, 2025]  
- **Tool/action planning error (wrong sequence/order)** (A2O/A2T, Semantic) [Ma, 2025]  
- **Non-determinism / run-to-run variance** (A2O/INFRA, Non-semantic) [Dong, 2024]  

---

# 5) Output / Format / Encoding Faults

- **Response formatting error (invalid JSON/schema)** (A2A/A2T, Non-semantic) [Ma, 2025; Deshpande, 2025]  
- **Language/encoding issue (symbols/encoding break parsers)** (A2A/A2T, Non-semantic) [Ma, 2025]  
- **Formatting errors as tool-call failures** (A2T, Non-semantic) [Deshpande, 2025]  

---

# 6) Termination & Verification Faults

- **Unaware of termination conditions** (A2O, Non-semantic control-flow) [Cemri, 2025]  
- **Premature termination** (A2O, Non-semantic control-flow) [Cemri, 2025]  
- **No / incomplete verification** (A2O, Semantic) [Cemri, 2025]  
- **Incorrect verification** (A2O, Semantic) [Cemri, 2025]  
- **Verification step removed/disabled (attack or config bug)** (A2O, Semantic) [Kong, 2025]  

---

# 7) Tool / API / Environment Execution Faults (Non-semantic primary)

## 7.1 Agentic tool/API errors (explicit in TRAIL + AgentFail)
- **Rate limiting (429)** (A2T, Non-semantic) [Deshpande, 2025]  
- **Authentication errors (401/403)** (A2T, Non-semantic) [Deshpande, 2025]  
- **Service errors (500)** (A2T, Non-semantic) [Deshpande, 2025]  
- **Resource not found (404)** (A2T, Non-semantic) [Deshpande, 2025]  
- **Tool invocation / KB retrieval error** (A2T, Non-semantic) [Ma, 2025]  
- **Tool definition/contract mismatch** (A2T, Non-semantic) [Deshpande, 2025]  
- **Environment setup errors (keys, permissions, missing deps)** (A2T/INFRA, Non-semantic) [Deshpande, 2025]  

## 7.2 Resource faults (agentic OS / sandbox / infra)
- **Resource exhaustion (memory overflow / overload)** (INFRA, Non-semantic) [Deshpande, 2025; Yu, 2025]  
- **Timeout issues (incl. infinite loops)** (A2T/INFRA, Non-semantic) [Deshpande, 2025]  

---

# 8) Platform / Infra Fault Models (Chaos + Microservices + DS)

## 8.1 Chaos experiments in LLM-MAS
- **Agent failures (agent crash/unavailable)** (INFRA, Non-semantic) [Owotogbe, 2025]  
- **Agent communication failures** (INFRA/A2A, Non-semantic) [Owotogbe, 2025]  
- **Communication delays** (INFRA/A2A, Non-semantic) [Owotogbe, 2025]  

## 8.2 Service-level fault injection (Filibuster)
- **Callsite exceptions (connection error, timeout)** (A2T, Non-semantic) [Meiklejohn, 2021]  
- **Injected HTTP error responses (e.g., InternalServerError, ServiceUnavailable)** (A2T, Non-semantic) [Meiklejohn, 2021]  

## 8.3 Distributed systems fuzzing model (Mallory)
- **Network partitions** (INFRA, Non-semantic) [Meng, 2023]  
- **Node faults (node crash/failure schedules)** (INFRA, Non-semantic) [Meng, 2023]  

## 8.4 Microservice FIT SLR (extra faults beyond classic crash/latency)
- **Scaling-operation failures (autoscaling side-effects)** (INFRA, Non-semantic) [Yu, 2025]  
- **Container image corruption** (INFRA, Non-semantic) [Yu, 2025]  
- **Configuration drift** (INFRA, Non-semantic) [Yu, 2025]  

---

# 9) Security / Adversarial Faults (Agentic-specific)

- **Direct prompt injection** (A2O, Semantic security) [Solomon, 2025; Bryan, 2025]  
- **Indirect prompt injection (via web/tool/RAG)** (A2T/A2M, Semantic security) [Solomon, 2025; Bryan, 2025; Li, 2024]  
- **Agent injection** (A2A/A2O, Semantic security) [Bryan, 2025]  
- **Agent impersonation** (A2A, Semantic security) [Bryan, 2025]  
- **Agent flow manipulation (routing/workflow tampering)** (A2O, Non-semantic → semantic harm) [Bryan, 2025]  
- **Agent provisioning poisoning (poison agent setup/config at birth)** (A2O, Semantic security) [Bryan, 2025]  
- **Multi-agent jailbreaks** (A2A/A2O, Semantic security) [Bryan, 2025]  
- **Human-in-the-loop bypass** (A2O/HITL, Semantic security) [Bryan, 2025]  
- **Function/tool compromise (malicious tool/function)** (A2T, Non-semantic+Semantic) [Bryan, 2025]  
- **Incorrect permissions / over-permissioning** (INFRA, Non-semantic security) [Bryan, 2025]  
- **Insufficient isolation (sandbox/tenant boundary issues)** (INFRA, Non-semantic security) [Bryan, 2025]  
- **Excessive agency (agent can take too-broad actions)** (A2O/INFRA, Semantic governance) [Bryan, 2025]  
- **Resource exhaustion attacks (DoS via loops/tool abuse)** (INFRA, Non-semantic security) [Bryan, 2025; Solomon, 2025]  

---

# 10) Observability / Ops-layer Faults (debuggability faults)

- **Missing trace linkage (broken parent-child, lost IDs)** (OBS, Non-semantic) [Dong, 2024]  
- **Missing artifact tracking (goals/plans/tools not recorded)** (OBS, Non-semantic) [Dong, 2024]  
- **Missing provenance / accountability records** (OBS, Non-semantic) [Dong, 2024; Bryan, 2025]  
- **Monitoring distortion (false latency/cost anomalies)** (OBS, Non-semantic) [Dong, 2024]  
- **Prompt/model version drift not recorded** (OBS, Non-semantic) [Dong, 2024]  

---

# 11) Failure Attribution / Counterfactual Fault Seeding (mechanism, but useful for injection design)

*(Not “new fault types”, but critical because they define injectable **where/when**.)*
- **Decisive-error step targeting: mistake = (agent i, time t)** (A2O, Semantic injection targeting) [S. Zhang, 2025]  
- **Intervention operator: replace action at step t to test causality** (A2O, Semantic injection operator) [S. Zhang, 2025]  

---

# 12) MODEL-level Faults (outside MAS layer, but in literature)

- **Bit-flip faults (memory/register faults during inference)** (MODEL, Non-semantic) [Yu, 2026]  
- **Weight/activation perturbation during forward pass** (MODEL, Non-semantic) [Yu, 2026]  
- **GPU/hardware FI (SEU-like failures)** (MODEL, Non-semantic) [Yu, 2026]  

---