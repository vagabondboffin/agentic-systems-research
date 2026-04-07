# Fault/Failure taxonomy for LLM-MAS (FI-oriented)

## Legend
- **Surface (primary)**:
  - `A2A` agent↔agent
  - `A2T` agent↔tool/KB/API
  - `WF` workflow/orchestrator (plans, routing, role logic)
  - `PLAT` platform/runtime/infra (env, deps, filesystem, network)
  - `DATA` data/KB + structured I/O contracts (schemas, parsing, encoding)
  - `MODEL` model/API layer (provider, auth, rate limits, capability)
- **Semantics**:
  - `Semantic` = meaning/intent/logic is corrupted
  - `Non-semantic` = timing/availability/format/transport/runtime issues

---

## Faults

### A2A — Semantic
- Communication instruction loss (CIL) {Jia,2026}
- Hallucinated message content / fabricated claims {Jia,2026}
- Blind trust in upstream messages (no cross-checking) {Jia,2026}

### A2A — Non-semantic
- Message delay / jitter {Jia,2026}
- Message drop/loss {Jia,2026}
- Message duplication (no dedup) {Jia,2026}
- Message cycles / replay storms {Jia,2026}
- Broadcast amplification / misrouting to unintended agents {Jia,2026}

---

### WF — Semantic
- Inexecutable plan / invalid decomposition {Cemri,2025}
- Instruction logic conflict / inconsistent constraints {Jia,2026}
- Instruction ambiguity (under-specified task) {Jia,2026}
- Role ambiguity / role confusion {Cemri,2025}
- Prompt formatting / specification error (poor structure cues) {Shah,2026}
- Prompt variable binding / orchestration error (missing variables, wrong structured input like string vs JSON) {Islam,2026}

### WF — Non-semantic
- Agent termination failure {Shah,2026}
- Agent execution failure (crash/halt at runtime) {Shah,2026}

---

### A2T — Semantic
- Tool selection error (wrong tool chosen) {Shah,2026}

### A2T — Non-semantic
- Tool format error (malformed tool output / contract violation) {Shah,2026}
- API misuse (wrong API context) {Shah,2026}
- API parameter mismatch (wrong/missing args) {Shah,2026}
- API misconfiguration {Shah,2026}

---

### MODEL — Semantic
- Poor prompt framing error (elicits structurally/semantically wrong outputs) {Shah,2026}
- Model capability mismatch (task requires capabilities the chosen model doesn’t support) {Islam,2026}

### MODEL — Non-semantic
- Token handling & tracking error {Shah,2026}
- LLM misconfiguration {Shah,2026}
- LLM usage / API incompatibility (schema drift, tool-call interface mismatch) {Shah,2026}
- LLM authentication failure {Shah,2026}
- Rate limiting / transient LLM API failure {Shah,2026}
- Model/service unavailable (downtime / feature not released / “model not found”) {Islam,2026}

---

### PLAT — Non-semantic
- Connection setup failure (infra/client setup) {Shah,2026}
- Authentication failure (external services) {Shah,2026}
- Authorization violation {Shah,2026}
- Resource handling error (files/locks/threads; FS unavailable mid-run) {Shah,2026}
- Database misconfiguration {Shah,2026}
- Memory persistence failure {Shah,2026}
- State load/save failure {Shah,2026}
- Import/reference resolution failure (deprecated/missing module path) {Islam,2026}
- Resource exhaustion / quota exhaustion (insufficient RAM/credits) {Islam,2026}

---

### DATA — Non-semantic
- Type handling error {Shah,2026}
- Logic/constraint violation in parsing/transforms {Shah,2026}
- Encoding/decoding error {Shah,2026}
- Validation omission {Shah,2026}
- File-type interpretation error {Shah,2026}
- LLM output parsing/schema mismatch (output format deviates from parser expectations) {Islam,2026}

---

## Failures

### System-level MAS failure modes (MAST)
- FM-1.1 Disobey task specification {Cemri,2025}
- FM-1.2 Disobey role specification {Cemri,2025}
- FM-1.3 Step repetition {Cemri,2025}
- FM-1.4 Loss of conversation history {Cemri,2025}
- FM-1.5 Unaware of termination conditions {Cemri,2025}
- FM-2.1 Conversation reset {Cemri,2025}
- FM-2.2 Fail to ask for clarification {Cemri,2025}
- FM-2.3 Task derailment {Cemri,2025}
- FM-2.4 Information withholding {Cemri,2025}
- FM-2.5 Ignored other agent’s input {Cemri,2025}
- FM-2.6 Reasoning–action mismatch {Cemri,2025}
- FM-3.1 Premature termination {Cemri,2025}
- FM-3.2 No or incomplete verification {Cemri,2025}
- FM-3.3 Incorrect verification {Cemri,2025}

---

### Observable failure symptoms (execution-level)
- Data & validation errors (schema/type mismatches, malformed outputs) {Shah,2026}
- Installation & dependency issues {Shah,2026}
- Execution & runtime failures (exceptions, crashes, hangs) {Shah,2026}
- Code quality & structure issues {Shah,2026}
- Agent-specific issues {Shah,2026}
- Error handling failures (missing/unclear recovery/logging) {Shah,2026}
- LLM-specific failures {Shah,2026}
- Connection & network errors {Shah,2026}
- Tool & function call issues {Shah,2026}
- File & resource errors {Shah,2026}
- Synchronization issues {Shah,2026}

#### Additional concrete “effects” (useful as FI outcome labels)
- Empty response (no output) {Islam,2026}
- Partial output (truncated/incomplete) {Islam,2026}
- Incorrect output (complete but wrong) {Islam,2026}
- Output dump (non-streaming “all at once” output) {Islam,2026}
- Tool ignored (tools not invoked when needed) {Islam,2026}
- Stateless interaction (memory not reflected; only current turn handled) {Islam,2026}
- Slow output {Islam,2026}
- Warning (non-terminating warning state) {Islam,2026}
- Indeterminate loop (infinite loop) {Islam,2026}
- Resource overuse (RAM/compute spike) {Islam,2026}
- Silent fail (task failed but no explicit indication) {Islam,2026}