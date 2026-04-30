# Davinci Out-of-Order Processor Core — v2

## 1. Overview & Design Philosophy

The Davinci-v2 core is a **single-threaded, 4-wide, out-of-order, speculatively-executing** processor targeting AI inference, HPC, and dense linear algebra workloads. It executes a unified instruction stream containing four instruction domains — scalar, vector, cube (matrix), and memory-tile-engine (MTE) — on a shared front-end with distributed back-end execution units.

**v2 inherits the v1 baseline ([`Davinci_supersclar.md`](Davinci_supersclar.md)) and adds three architectural enhancements:**

1. **TRegFile-4K with per-port `is_transpose` reads** (see [`tregfile4k.md`](tregfile4k.md) §7). Each read port now carries a 1-bit `is_transpose` flag latched at the epoch boundary alongside `reg_idx`. When asserted, the port delivers the **chunk-grid transpose** of the addressed 4 KB tile at full **512 B/cy** for the entire 8-cycle epoch — bank-conflict-free under the diagonal skew, with no SRAM duplication, no extra latency, and no dedicated transpose buffer. This eliminates the v1 `TILE.TRANSPOSE` predecessor for most use cases and enables the vector unit's per-beat tilelet-transpose mechanism (§8.3).
2. **Vector unit upgraded to VEC-4K-v2** (see [`vector4k_v2.md`](vector4k_v2.md)). Major changes vs. v1's vector unit:
   - Up to **3 tile operands** per instruction: two value tiles (`A`, `B`) plus one **per-element bitmask tile** (`C`).
   - Up to **2 tile results** per instruction (`D0`, `D1`) — value+index, quotient+remainder, etc.
   - **Per-element predication / masking** on every elementwise op, every reduction, and every gather/sort — at zero fetch-phase cost in the common case.
   - **Tile-register metadata** (32 b: `shape.x`, `shape.y`, `format`) carried alongside each 4 KB tile.
   - **SRAM-based staging registers** (`SA`, `SB`, `SC`) decouple TRegFile fetch cadence from compute pipeline; per-beat microcode dispatches `{src, strip, tilelet_xpose}` per ALU operand.
   - **Restored narrow formats**: FP8 (E4M3/E5M2), FP4 (MXFP4/HiFP4) joining FP32, FP16, BF16.
   - **Three new PTO instructions** natively enabled by the v2 datapath: `TINV` (matrix inverse up to 128×128 FP32 / 16-tile range), `TROWRANGE_MUL` (column-wise product over a dynamic row sub-range), `TMRGSORT` (full-tile mergesort over any `N = 2^p` up to 8192 via a reconfigurable 256-lane shuffle + compare-swap primitive).
3. **Speculative out-of-order execution** with a **ROB-less recovery scheme** that nonetheless guarantees architectural state is never corrupted by a misspeculated path (§11). The mechanism extends the v1 RAT-checkpoint + reference-counting infrastructure with a **branch-tagged speculative store buffer** for scalar memory and a **speculative tile-store queue** for MTE memory writes — both of which gate visible side effects until the producing branch tag becomes non-speculative. Section 11 walks through why this is sufficient without a Reorder Buffer, what it costs in area / latency, and which workloads it can and cannot serve correctly.

> **Design discipline:** The v2 core continues to assume **run-to-completion kernel execution** with **no precise architectural exceptions and no OS-level interrupts** — the same envelope as v1. The new speculation-recovery mechanism handles **branch mispredictions** only; it does **not** turn the core into a precise-exception machine. Section 11.7 enumerates the remaining "non-recoverable" classes (asynchronous page faults, signaling NaNs, ECC errors observed mid-kernel) and the kernel-level conventions that bound them.

### 1.1 Key Parameters (v2 deltas in **bold**)

| Parameter | Value |
|-----------|-------|
| Scalar ISA width | **64-bit** RISC (ARM / RISC-V style), unchanged |
| Architectural GPRs | **32** (X0–X31), 64-bit |
| Physical GPRs | **128** (P0–P127), 64-bit |
| Architectural tile regs | **32** (T0–T31), 4 KB each |
| Physical tile regs | **256** (PT0–PT255) in TRegFile-4K |
| **TRegFile-4K read ports** | **8R, each with `is_transpose` bit** (§9.2) |
| **Per-tile metadata** | **32 b** (shape.x, shape.y, format) §9.2.1 |
| Fetch / decode width | **4** instructions / cycle |
| Scalar issue width | **6** (4 ALU + 1 MUL/DIV + 1 branch) |
| **Vector issue width** | **1 VEC-4K-v2 instruction / cycle** (§8.3) |
| Cube issue width | **1** CUBE instruction / cycle |
| MTE issue width | **2** TILE.LD/ST per cycle |
| Pipeline depth (scalar) | **12** stages (fetch-to-writeback) |
| Branch predictor | Hybrid TAGE + BTB + RAS |
| **RAT checkpoints** | **8** (max in-flight branches; same as v1, now repurposed as *speculation tags* §11.3) |
| **Branch tag width** | **3 b** (matches checkpoint count); attached to every in-flight RS / store-buffer / tile-store-queue entry |
| Reservation station entries | Scalar: 32, LSU: 24, **Vector: 24** (was 16; widened for 3-operand v2 entries), Cube: 4, MTE: 16 |
| **Speculative store buffer entries** | **24** (was 16 in v1; widened to absorb branch-tag gating §11.4) |
| **Speculative tile-store queue** | **8** entries (branch-tag-gated, MTE-side §11.5) |
| L1-I cache | 64 KB, 4-way, 64 B line |
| L1-D cache | 64 KB, 4-way, 64 B line, non-blocking (8 MSHRs) |
| L2 cache (core-private) | 512 KB, 8-way, 64 B line |
| Cube MXU | 4096 base MACs, 8 banks, dual-mode A/B |
| Clock target | ≥ **1.5 GHz** (5 nm) |
| **Peak FP32 throughput (vec)** | **0.77 TFLOPS** (1 tile / 8 cy at 1.5 GHz, 128-lane FMA) |
| **Peak FP4 throughput (vec)** | **6.14 TFLOPS** (4× SIMD per group) |
| Peak FP16 throughput (cube) | **12.3 TFLOPS** |
| Peak FP8 throughput (cube) | **24.6 TOPS** |
| Peak MXFP4 throughput (cube) | **98.3 TOPS** |

---

## 2. ISA Summary

The v2 ISA is a strict superset of v1: every v1 opcode encodes identically and behaves identically. v2 adds:

- **Masked variants** of every elementwise vector op, every reduction, and every gather (encoded by a bit in `funct7`).
- **Three new PTO instructions** (§2.2.6).
- **A new tile-metadata setter** `TSETMETA` (§2.2.7).
- **Branch hint bits** in the conditional-branch encoding for static prediction override (§5.2.4).

### 2.1 Scalar ISA

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §2.1。)**

A 64-bit RISC instruction set with ARM / RISC-V style operations.

| Category | Instructions | Operands | Latency (cycles) |
|----------|-------------|----------|-------------------|
| Integer ALU | ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, MOV | 2 src GPR, 1 dst GPR | 1 |
| Immediate ALU | ADDI, ANDI, ORI, XORI, SLLI, SRLI, SRAI, LUI | 1 src GPR + imm, 1 dst GPR | 1 |
| Multiply | MUL, MULH, MULHU | 2 src GPR, 1 dst GPR | 4 (pipelined) |
| Divide | DIV, DIVU, REM, REMU | 2 src GPR, 1 dst GPR | 12–20 (non-pipelined) |
| Compare & branch | BEQ, BNE, BLT, BGE, BLTU, BGEU | 2 src GPR + offset | 1 (resolve) |
| Jump | JAL, JALR | 1 src GPR + offset, 1 dst GPR | 1 |
| Load | LB, LH, LW, LD, LBU, LHU, LWU | 1 src GPR + offset, 1 dst GPR | 4 (L1 hit) |
| Store | SB, SH, SW, SD | 2 src GPR + offset | 4 (L1 hit) |
| System | FENCE, NOP, HALT | — | varies |

**Architectural registers:** X0 (hardwired zero) through X31, plus a program counter (PC). Condition flags are not used; branches compare register values directly (RISC-V style).

**Encoding (32-bit):**

```
  31       25 24  20 19  15 14  12 11   7 6     0
 ┌──────────┬──────┬──────┬──────┬──────┬────────┐
 │  funct7  │  rs2 │  rs1 │funct3│  rd  │ opcode │  R-type
 └──────────┴──────┴──────┴──────┴──────┴────────┘

 ┌─────────────────┬──────┬──────┬──────┬────────┐
 │    imm[11:0]    │  rs1 │funct3│  rd  │ opcode │  I-type
 └─────────────────┴──────┴──────┴──────┴────────┘
```

**v2 增量 (§5.2.4):** A new optional 1-bit `H` (hint) field in the conditional-branch funct3 encoding lets the compiler suggest static taken/not-taken when the dynamic predictor has no entry. Predictor still has final say once it has trained — H is consulted only on a TAGE/BTB miss. v1 software runs on v2 unmodified — the `H` bit defaults to 0 (no hint) when assembled by a v1-targeted compiler.

### 2.2 Vector ISA — VEC-4K-v2

The vector unit consumes **4 KB tile registers** (`T0–T31`, renamed by the Tile RAT to physical tile slots `PT0–PT255` in TRegFile-4K). All vector instructions are 32-bit fixed-width, with the same R/S/T/U-type encoding skeleton as v1 (§2.2.2 of v1).

**v2 changes:**

#### 2.2.1 Tile metadata (`shape.x`, `shape.y`, `format`)

Every physical tile register carries a **32-bit metadata word** alongside its 4 KB payload (§9.2.1):

```
  ┌────────────┬────────────┬───────────┬─────────────────────┐
  │ shape.x    │ shape.y    │ format    │ flags / reserved    │
  │ [13:0]     │ [27:14]    │ [31:28]   │                     │
  └────────────┴────────────┴───────────┴─────────────────────┘
```

| Field | Width | Range | Meaning |
|-------|-------|-------|---------|
| `shape.x` | 14 b | 1 … 8192 | Number of **columns `C`** (logical row length). Power-of-two only. |
| `shape.y` | 14 b | 1 … 8192 | Number of **rows `R`**. Power-of-two only. |
| `format`  | 4 b  | see below | Logical element format. |
| `flags`   | (overlay) | — | `arg_tile`, `scalar_tile`, `prefetch_hint` (microcode-encoded). |

Legality: `shape.x · shape.y · E = 4096`, where `E` is bytes/element from the format table:

| `format` code | Logical name | `E` (bytes) | Elements/tile |
|----------------|--------------|-------------|---------------|
| `0b0000` | FP32 / INT32 | 4 | 1024 |
| `0b0001` | FP16 / BF16  | 2 | 2048 |
| `0b0010` | FP8 (E4M3 / E5M2) | 1 | 4096 |
| `0b0011` | FP4 (MXFP4 / HiFP4) | 0.5 | 8192 |
| `0b01xx`–`0b11xx` | reserved | — | — |

Metadata is written **only** by the producing instruction (implicit at retire) or the explicit `TSETMETA` op (§2.2.7). It cannot change while the tile is the source of any in-flight fetch.

> **Why metadata?** It lets a single tile op service every shape/format without opcode explosion. The vector unit's stage (A) align/unpack (§8.3) and stage (B) reduction (§8.3) both consult `format` and `shape` from the tile-metadata word read at the first strip of each operand. Microcode programs (§8.3.4) are keyed by `(opcode, format, W-regime, R-regime)`.

#### 2.2.2 Operand model (3 source, 2 destination)

| Operand | Role | Tile-RAT entry | Storage |
|---------|------|----------------|---------|
| **A** | Value tile (primary, mandatory) | source | TRegFile read port R0 → `SA` staging |
| **B** | Value tile (secondary, optional) | source | TRegFile read port R4 → `SB` staging |
| **C** | **Dual role:** `c_role = MASK` → per-element bitmask (1 b/element); `c_role = VALUE` → **third value tile** for native 3-source FMA family (§2.2.6a) | source (when `has_mask = 1` **or** `c_role = VALUE`) | TRegFile read port R1 (v2.1: 3rd VEC-side binding) → `SC` staging |
| **D0** | Result tile (primary) | destination | Write port `W0` |
| **D1** | Result tile (secondary, optional) | destination | Write port `W4` |

The 32-bit instruction word reserves:
- a `c_role` bit (0 = `MASK`, 1 = `VALUE`),
- a `has_mask` bit (1 if `C` is fetched **and** `c_role = MASK`),
- a `retire_mask[1:0]` field (which of `D0`, `D1` are written), and
- per-operand `is_transpose_{A,B,C}` bits forwarded to the TRegFile read ports (§9.2).

Tile register fields stay 5 bits (T0–T31). When `c_role = VALUE` and `N_val = 3` (e.g. `VFMA`, `VFNMA`, `VLERP`), `C` is fetched as a full 4 KB value tile through the dedicated VEC read port R1 — see §2.2.6a and [`vector4k_v2.md`](vector4k_v2.md) §3.1, §7.6 for the rationale and 3-port binding.

> **Why a third VEC-side TRegFile read port?** TRegFile-4K has 8 physical read ports. v1 and v2.0 used only 2 (R0/R4) for VEC, since operand `C` was strictly a small mask. v2.1 binds **R1 = Port C** so that all three value tiles of a 3-source `VFMA` can be fetched **in parallel within one 8 cy epoch** — same cadence as a binary op. The alternative (sequential 2-epoch fetch on R0/R4) would halve `VFMA` throughput. Bandwidth cost: 0 SRAM, 0 bank-conflict pressure (the diagonal skew already supports 8 conflict-free read ports per [`tregfile4k.md`](tregfile4k.md) §4); only a binding allocation. R1 is idle and clock-gated when no `c_role = VALUE` op is in flight.

#### 2.2.3 Encoding (32-bit)

```
  R-type (3-source, 2-dest):
  31      26 25  21 20  16 15   12 11      6 5     0
 ┌────────┬──────┬──────┬──────┬───────┬────────┐
 │ funct6 │  Tc  │  Tb  │  Ta  │ Td0/d1│ opcode │
 │ + xpA  │ (5b) │ (5b) │ (5b) │ (6b)  │ VEC    │
 │ + xpB  │      │      │      │       │        │
 │ + xpC  │      │      │      │       │        │
 │ +mask  │      │      │      │       │        │
 │ +crole │      │      │      │       │        │  ← v2.1: c_role bit (MASK/VALUE)
 │ +rmask │      │      │      │       │        │
 └────────┴──────┴──────┴──────┴───────┴────────┘
   funct6 packs 6 bits split between op-extension (3 b),
   has_mask (1 b), is_xpose_A (1 b), is_xpose_B (1 b);
   is_xpose_C, c_role, and retire_mask travel in the
   immediate slot of S-/T-types or in a fixed funct7
   bit pattern.
```

**Backward compatibility:** v1 vector instructions decode as `has_mask = 0`, `c_role = MASK`, `retire_mask = 2'b01`, `is_xpose_{A,B,C} = 0` — i.e. unmasked, single-result, no-transpose, no-3rd-tile — and produce bit-exact v1 results. `c_role = VALUE` is only generated by a v2.1-aware compiler emitting `VFMA` / `VFNMA` / `VLERP`; v1 binaries cannot express it.

#### 2.2.4 Masked / predicated variants

Every elementwise op (`VADD`, `VMUL`, `VFMA`, `VSEL`, …), every reduction (`VROWSUM`, `VCOLMAX`, …), and every gather (`VGATHER`, `VGATHERB`) has a masked counterpart. Conventions:

| Encoding | Behaviour |
|----------|-----------|
| `has_mask = 0` | Full-tile op; `C` not fetched; lane gate ties to `IMM_ALL_ONES` (every lane participates). Identical to v1 semantics. |
| `has_mask = 1` | `C` fetched (≤ 2 strips piggybacking on an idle read-port cycle, §8.3.6); per-lane gate `out[lane] = M[lane] ? alu_core_out[lane] : identity[lane]`. The "identity" depends on the op (see [`vector4k_v2.md`](vector4k_v2.md) §5.8): preserves operand A for `TSEL`-style, leaves accumulator unchanged for masked `ACCUM`, etc. |

**Cycle cost.** Masked variants pay **0 extra fetch cycles** in the common case (mask piggybacks on idle port within the value-tile epoch). End-to-end latency is therefore the same as the unmasked variant.

#### 2.2.5 Per-operand transpose (`is_xpose_*`) and per-beat tilelet transpose

Two orthogonal transpose mechanisms, both reusing the chunk-grid transpose algorithm at 64 B sub-chunk granularity:

| Mechanism | Granularity | Control | Where |
|-----------|-------------|---------|-------|
| **TRegFile read-port `is_transpose`** | Whole tile, set once at fetch time | `is_xpose_{A,B,C}` bits in the issue packet | TRegFile-4K read port ([`tregfile4k.md`](tregfile4k.md) §7) — costs zero VEC-side hardware |
| **Staging-side `tilelet_xpose`** | Per 512 B tilelet, per-beat | One bit per operand slot in each microcode beat | Staging register (`SA`, `SB`, `SC`) read datapath — costs ~30 K gate per staging register |

The two combine: a tile may be fetched in row-mode and then read in col-mode beat-by-beat from staging, or fetched pre-transposed (col-mode) and re-read in row-mode. Microcode picks the combination per instruction.

**TRegFile-side rule R2** ([`tregfile4k.md`](tregfile4k.md) §6): the two physical read ports active in any 8-cycle epoch must share the same `is_transpose`. For a 2-value-operand instruction with `is_xpose_A ≠ is_xpose_B`, microcode splits the fetch into two epochs (16 cy instead of 8 cy) — this is the only scheduling cost of the new flag (§8.3.6).

#### 2.2.6a Native 3-source ternary FMA family (`VFMA`, `VFNMA`, `VLERP`) — v2.1 增量

A new family of vector instructions that consume **three independent value tiles** and produce one (or two) result tiles, enabled by the operand-`C` dual-role mechanism (§2.2.2) and the 3rd VEC-side TRegFile read port (R1, [`vector4k_v2.md`](vector4k_v2.md) §3.1, §7.6). v0.16 of [`vector4k_v2.md`](vector4k_v2.md) only supported ternary FMA via the **accumulator feedback path** (`VFMA_ACC D = A·B + Acc`), which is suitable for GEMM-epilogue / FMA-accumulate kernels but **fails** for the canonical FMA pattern `D = A·B + C` where the third operand is **not** the previous accumulator.

| Mnemonic | Operands | Semantics | Encoding | Cycle budget (typical, uniform `is_transpose`) |
|----------|----------|-----------|----------|-------------------------------------------------|
| **VFMA** | `Td0, Ta, Tb, Tc` | `Td0 = Ta · Tb + Tc` (single-rounding IEEE-754 FMA, all formats) | R-type, `c_role = VALUE`, `has_mask = 0` | **8 cy fetch (3-port parallel) + 8 compute beats + 1 cy fall-through ≈ 10–12 cy end-to-end** — same as `VADD`/`VMUL` |
| **VFNMA** | `Td0, Ta, Tb, Tc` | `Td0 = -(Ta · Tb) + Tc` | R-type, `c_role = VALUE`, `funct6.fnma = 1` | same as `VFMA` |
| **VLERP** | `Td0 [, Td1], Ta, Tb, Tc` | `Td0 = Ta · (1 − Tc) + Tb · Tc` (linear interpolation; optional `Td1 = Tb − Ta` retired in same op) | R-type, `c_role = VALUE`, `funct6.lerp = 1` | 2 fused beats per strip (8 strip × 2 beat = 16 compute beats), `~18 cy` end-to-end |

**Why this family is needed.** From [`FMA指令场景说明.md`](FMA指令场景说明.md):

| Real-world kernel | FMA form | Notes |
|-------------------|----------|-------|
| **LayerNorm / RMSNorm final affine** | `y = γ·x̂ + β` | The dominant FMA in transformer normalisation. `γ`, `x̂`, `β` are three independent tile registers — none is the accumulator. Without `VFMA`, every LayerNorm pays 2× cost (`VMUL` + `VADD`). |
| **Welford incremental update — mean** | `μ_new = δ·inv_n + μ_old` | Streaming variance estimator at the heart of LayerNorm reductions. |
| **Welford incremental update — M2** | `M2_new = δ·δ_2 + M2_old` | Single-rounding FMA preserves precision against catastrophic cancellation on small variance terms (matters for FP16 / BF16 / FP8). |
| **Welford state merge** | `μ = δ·factor + μ_A`; `M2 = M2_A + δ·(δ·factor_m2) + M2_B` | Distributed-norm cross-thread merges. |
| **Activation polynomials** | `gelu`, `swiglu` polynomial / Padé approximations | Multiple FMAs over independent tile inputs. |
| **Trigonometric polynomials** | `sin(x) ≈ x·(c₁ + x²·(c₃ + x²·c₅))` | Horner-form FMAs. |

**Justification (decisive advantages of FMA over emulated `MUL` + `ADD`):**

1. **Throughput doubling** — one fused instruction instead of two halves the FMA-bound pipeline depth and the issue/RS occupancy.
2. **Precision preservation** — IEEE-754 FMA performs a *single* rounding after the infinite-precision `A·B` intermediate, eliminating the second-rounding error of `(A·B) + C`. This matters for FP16 / BF16 / FP8 normalisation kernels that re-feed the result into subsequent reductions.

**Hardware delta (vs. v0.16).** The stage (B) per-lane FMA core, microcode beat machinery, and 8-port TRegFile already supported `A·B + Z`. The only structural changes are:

| Block | Δ |
|-------|---|
| `MUX_Z` per-lane input MUX (already 6:1) — one source retargeted to `SC` value-mode read | ~0 (same gate count) |
| `SC` staging — add 512 B/cy value-mode read path alongside the existing 1-bit-mask read path (sub-bank tree reused from [`vector4k_v2.md`](vector4k_v2.md) §4.2.1) | **~5 K gate** |
| TRegFile read port R1 binding to VEC | **0** (allocation only — TRegFile-4K already has 8R) |
| Issue-time `c_role` bit through Tile RAT / RS / dispatch | **~1 K gate** (control-path widening) |
| **Total v2.1 hardware add** | **~6 K gate (~0.2 % of VEC-4K-v2 area)** |

**Pipeline timing** (Davinci-v2.1 vector pipeline, [`vector4k_v2.md`](vector4k_v2.md) §6.2):

| Op | `N_val` | `c_role` | `is_transpose` mix | Fetch | Compute | End-to-end | Throughput |
|----|--------:|----------|---------------------|------:|--------:|-----------:|------------|
| **VFMA** (typical) | 3 | VALUE | uniform | **8 cy** | 8 beats | **~10–12 cy** | **1 tile / 8 cy** |
| VFMA (one xp odd-out) | 3 | VALUE | one-mismatched | 16 cy | 8 beats | ~18 cy | 1 tile / 16 cy |
| VFMA (all xp different — degenerate) | 3 | VALUE | all distinct | 24 cy | 8 beats | ~26 cy | 1 tile / 24 cy |

**Backward compatibility.** A v1 / v2.0 binary emits `c_role = MASK` exclusively; the new instructions are decoded only when the v2.1-aware compiler sets `c_role = VALUE`. Old binaries see no behaviour change and the R1 read port stays idle and clock-gated.

#### 2.2.6 New PTO instructions

Three instructions native to v2's unified ALU + Acc feedback + microcode pipeline ([`vector4k_v2.md`](vector4k_v2.md) §7.5):

| Mnemonic | Operands | Semantics | Cycle budget |
|----------|----------|-----------|--------------|
| **TINV** | `Tdst+, Tsrc+, num_tiles` | Square matrix inverse via in-tile Gauss–Jordan with Newton–Raphson reciprocal refinement. Up to **128×128 FP32 (16 tiles)**, 64×64 FP8 (1 tile), 32×32 FP32 (1 tile). | ≈ 2·N²·`S_row` + N·`S_col` + 3N beats. **33 K beats / ~33 µs for 128×128 FP32 @ 1 GHz**. |
| **TROWRANGE_MUL** | `Tdst, Tsrc, Xstart, Xend [, Tmask]` | Column-wise product over dynamic row sub-range `[Xstart, Xend)`. `out[c] = ∏_{r=Xstart}^{Xend−1} Tsrc[r, c]`. Optional mask further filters elements. | `1 + S_active + 1 ≤ 10` beats; ~18 cy end-to-end. |
| **TMRGSORT** | `Td0 (values), Td1 (indices), Tsrc, N` | Full-tile bitonic sort over any `N = 2^p` up to **8192** (FP4 tile). Emits sorted values to `D0` and permutation indices to `D1`. Optional mask = partial-sort. | `p(p+1)/2 × ⌈N/256⌉` beats. **220 beats for N=1024 FP32, 36 beats for N=256.** |

Additional encoding notes:

- **TINV multi-tile** uses a 2-bit `log₂(num_tiles)` field in `funct7` to select `num_tiles ∈ {1, 2, 4, 8, 16}`. Operand register fields then encode the **base** tile of each consecutive range.
- **TROWRANGE_MUL** sources `Xstart`, `Xend` from scalar registers via the staging slots `SX`, `SY` ([`vector4k_v2.md`](vector4k_v2.md) §4.3) — these are read at issue time from the scalar GPR file, costing **0 vector-side cycles**.
- **TMRGSORT** uses `N` from a 4-bit immediate field encoding `log₂(N) ∈ {5..13}` (32..8192).

#### 2.2.7 `TSETMETA` (tile metadata setter)

```
  TSETMETA Td, shape.x_imm, shape.y_imm, format_imm
```

A single-cycle, tile-RAT-only instruction that **rewrites the metadata word** of the destination tile's *current* physical mapping without touching its 4 KB payload. Handled at the D2 rename stage similarly to `TILE.MOVE`: no RS entry, no execute stage. The new metadata becomes visible to subsequent instructions consuming `Td`.

Use cases: reshaping a tile produced by `CUBE.DRAIN` (which writes payload but not shape), changing format after `VCVT`, or installing scalar-broadcast metadata before a `VEXPAND`.

#### 2.2.8 Updated vector instruction list (highlights)

The **95-instruction v1 vector ISA** carries forward, with three changes:

1. Each instruction gets a "masked" variant (no new mnemonic — encoded by `has_mask`).
2. `TSORT32` and `TMRGSORT` from v1 are subsumed by the new `TMRGSORT` (§2.2.6). v1's `TSORT32` mnemonic remains as an alias for `TMRGSORT N=32`.
3. **v2.1 增量:** A new family of native 3-source ternary FMA instructions (`VFMA`, `VFNMA`, `VLERP`) is added under Category O (§2.2.6a), motivated by LayerNorm / Welford / activation / trig kernels (see [`FMA指令场景说明.md`](FMA指令场景说明.md)).

Categories A–M of v1 §2.2.3 are unchanged in semantics. Two new categories are added:

**Category N — Numerical / Reconfigurable Compute (new in v2)**

| Mnemonic | Operands | Semantics | Latency |
|----------|----------|-----------|---------|
| TINV | Tdst+, Tsrc+, num_tiles | Matrix inverse (Gauss–Jordan + NR refine) | ~2 K – 33 K beats |
| TROWRANGE_MUL | Tdst, Tsrc, Xstart, Xend [, Tmask] | Range product per column | ≤ 10 beats |
| TMRGSORT | Td0, Td1, Tsrc, log2N [, Tmask] | Bitonic sort, value+index dual retire | 36 – 2 912 beats |
| TSETMETA | Td, shape.x, shape.y, format | Rewrite tile metadata in-place | 0 (rename-only) |

**Category O — Native 3-source Ternary FMA family (new in v2.1; §2.2.6a)**

| Mnemonic | Operands | Semantics | Latency (typical) |
|----------|----------|-----------|--------------------|
| **VFMA** | `Td0, Ta, Tb, Tc` | `Td0 = Ta · Tb + Tc` (single-rounding IEEE-754 FMA) | **~10–12 cy** end-to-end (8 cy fetch + 8 compute beats); throughput **1 tile / 8 cy** |
| **VFNMA** | `Td0, Ta, Tb, Tc` | `Td0 = -(Ta · Tb) + Tc` | same as `VFMA` |
| **VLERP** | `Td0 [, Td1], Ta, Tb, Tc` | `Td0 = Ta·(1−Tc) + Tb·Tc`; optional `Td1 = Tb − Ta` retired in same instruction | ~18 cy end-to-end (8 cy fetch + 16 compute beats) |

All three issue with `c_role = VALUE`. Mixed `is_transpose_{A,B,C}` adds 8 cy per odd-out (one mismatch → 16 cy fetch; all three different → 24 cy fetch — degenerate). Common kernels (LayerNorm `γ·x̂ + β`, Welford updates) all use uniform `is_transpose` and hit the **8 cy** path.

### 2.3 Cube ISA

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §2.3。)**

Tile-level instructions that drive the outerCube MXU. Each `CUBE.OPA` consumes tile registers and executes all K-loop OPA steps internally.

| Instruction | Operands | Function |
|-------------|----------|----------|
| CUBE.CFG | mode, fmt [, Mactive] | Set operating mode (A/B) and data format |
| CUBE.OPA | zd, Ta, Tb, Rn | Outer product accumulate: iterate over Nb B-tiles |
| CUBE.DRAIN | zd, Tc | Drain accumulator buffer to tile register(s) |
| CUBE.ZERO | zd | Zero accumulator buffer (1 cycle) |
| CUBE.WAIT | zd | Stall until pending drain completes |

Supported formats: FP16, BF16, FP8 (E4M3/E5M2), MXFP4, HiFP4. All accumulate into FP32.

Full cube ISA specification: see [`outerCube.md`](outerCube.md) §6. The Tile RAT renames cube operands exactly as in v1; the outerCube MXU itself is unmodified between v1 and v2.

### 2.4 MTE ISA (Memory Tile Engine)

> **(v1 → v2: ISA 编码与语义 100% 兼容。以下 §2.4.1 / §2.4.2 / §2.4.3 完整复制自 v1 §2.4。v2 实现层增量列在本节末。)**

The MTE bridges three domains: **memory ↔ TRegFile-4K** (bulk tile transfers) and **scalar GPR ↔ TRegFile-4K** (single-element access). All MTE instructions flow through both the Scalar RAT and Tile RAT at rename.

#### 2.4.1 Bulk Tile Transfer Instructions

| Instruction | Operands | Function |
|-------------|----------|----------|
| TILE.LD | Td, [Rbase] | Contiguous load: 4 KB from address Rbase → tile Td |
| TILE.LD | Td, [Rbase], Rs | Strided load: rows at stride Rs → tile Td |
| TILE.ST | [Rbase], Ts | Contiguous store: tile Ts → 4 KB at address Rbase |
| TILE.ST | [Rbase], Ts, Rs | Strided store: tile Ts → rows at stride Rs |
| TILE.GATHER | Td, [Rbase], Tidx | Gather: indexed load using index tile (element offsets in Tidx) |
| TILE.SCATTER | [Rbase], Ts, Tidx | Scatter: indexed store using index tile (element offsets in Tidx) |
| TILE.ZERO | Td | Zero tile register Td |
| TILE.COPY | Td, Ts | Copy tile Ts → Td (allocates new physical tile, copies data) |

#### 2.4.2 Tile Manipulation Instructions

| Instruction | Operands | Function |
|-------------|----------|----------|
| TILE.MOVE | Td, Ts | Move tile Ts → Td (rename-only, zero-copy; see move elimination below) |
| TILE.TRANSPOSE | Td, Ts, fmt | Transpose tile Ts with element format fmt → tile Td |

**TILE.MOVE Td, Ts** — Logically copies tile Ts to Td, but is implemented as **move elimination** at the rename stage: the Tile RAT entry for Td is simply updated to point to the same physical tile as Ts. No data is copied, no physical tile is allocated from the free list, and no execute stage is needed. The instruction completes in **zero cycles** (handled entirely at D2 rename).

```
  Rename (D2) for TILE.MOVE Td, Ts:
    1. Read Tile RAT[Ts] → PT_src (current physical tile for Ts)
    2. Read Tile RAT[Td] → PT_old (old physical tile for Td, becomes orphan)
    3. Write Tile RAT[Td] ← PT_src  (Td now aliases same physical tile as Ts)
    4. Increment refcount(PT_src)     (one more architectural name maps to it)
    5. Mark PT_old as orphan; if refcount(PT_old)==0 → free to tile free list
    6. No RS entry allocated; no execute stage; instruction retires at D2
    7. Ready bit for Td inherits ready state of PT_src
```

After TILE.MOVE, Td and Ts share the same physical tile. This is safe under rename: the next instruction that writes to either Td or Ts will allocate a fresh physical tile at that point, naturally "splitting" the alias. TILE.MOVE is critical for avoiding unnecessary 4 KB copies in tile register spill/fill sequences and data routing between pipeline stages.

**TILE.TRANSPOSE Td, Ts, fmt** — Reads tile Ts, transposes the 2D element matrix according to the element format `fmt`, and writes the result to tile Td. The transpose treats the 4 KB tile as a 2D matrix with dimensions determined by the element width:

| fmt (funct3) | Element width | Tile layout (rows × cols) | Transpose block |
|-------------|--------------|---------------------------|-----------------|
| 000 (FP64) | 8 B | 64 × 8 | 8 × 8 (8 blocks of 8 rows) |
| 001 (FP32) | 4 B | 64 × 16 | 16 × 16 (4 blocks of 16 rows) |
| 010 (FP16) | 2 B | 64 × 32 | 32 × 32 (2 blocks of 32 rows) |
| 011 (BF16) | 2 B | 64 × 32 | 32 × 32 (2 blocks of 32 rows) |
| 100 (FP8) | 1 B | 64 × 64 | 64 × 64 (1 block, full tile) |
| 101 (INT32) | 4 B | 64 × 16 | 16 × 16 (4 blocks of 16 rows) |
| 110 (INT16) | 2 B | 64 × 32 | 32 × 32 (2 blocks of 32 rows) |
| 111 (INT8) | 1 B | 64 × 64 | 64 × 64 (1 block, full tile) |

The transpose operates on **square sub-blocks** whose dimension equals the number of elements per 512-bit row. For FP8/INT8 the entire tile is one 64×64 block and transposes in-place. For FP16/BF16/INT16, the 64 rows are split into two 32-row halves, each transposed as a 32×32 block. In v1, the MTE unit contained a dedicated **transpose buffer** (4 KB SRAM) that accumulated rows during the read epoch and emitted transposed rows during the write epoch. In v2 this buffer shrinks to 512 B (see §8.5.1).

```
  TILE.TRANSPOSE encoding (32-bit):
  ┌──────────┬──────┬──────┬──────┬──────┬────────┐
  │  funct7  │ 00000│  Ts  │ fmt  │  Td  │ opcode │
  │ 0100010  │ (5b) │ (5b) │(3b)  │ (5b) │ 10xxxxx│
  └──────────┴──────┴──────┴──────┴──────┴────────┘
```

#### 2.4.3 Scalar ↔ Tile Element Access Instructions

| Instruction | Operands | Function |
|-------------|----------|----------|
| TILE.GET | Rd, Ts, Ridx | Read single element: element at index Ridx in tile Ts → scalar GPR Rd |
| TILE.PUT | Td, Rs, Ridx | Write single element: scalar GPR Rs → element at index Ridx in tile Td |

**TILE.GET Rd, Ts, Ridx** — Reads one element from tile Ts at the position specified by scalar register Ridx. The element is zero-extended to 64 bits and written to scalar destination GPR Rd. The element data type (FP16, FP32, FP64, INT8, etc.) is encoded in the instruction's `funct3` field, which determines element width and the extraction offset within the 512-bit row. Ridx encodes a linear element index: `row = Ridx / elements_per_row`, `col = Ridx % elements_per_row`.

**TILE.PUT Td, Rs, Ridx** — Writes the lower bits of scalar GPR Rs into tile Td at the element position specified by Ridx. This is a **read-modify-write** operation on the tile: the rename stage treats Td as both source (old mapping, read) and destination (new physical tile, write). The MTE unit copies the source physical tile to the destination physical tile, then overwrites the single element. The element data type is encoded in `funct3`.

```
  TILE.GET encoding (32-bit):
  ┌──────────┬──────┬──────┬──────┬──────┬────────┐
  │  funct7  │ Ridx │  Ts  │funct3│  Rd  │ opcode │
  │ 0100000  │ (5b) │ (5b) │ type │ (5b) │ 10xxxxx│
  └──────────┴──────┴──────┴──────┴──────┴────────┘
       Ts: architectural tile register (T0–T31)
       Ridx: scalar GPR holding element index
       Rd: scalar GPR destination
       funct3: element type (000=FP64, 001=FP32, 010=FP16, 011=BF16, 100=FP8, 101=INT32, 110=INT16, 111=INT8)

  TILE.PUT encoding (32-bit):
  ┌──────────┬──────┬──────┬──────┬──────┬────────┐
  │  funct7  │  Rs  │ Ridx │funct3│  Td  │ opcode │
  │ 0100001  │ (5b) │ (5b) │ type │ (5b) │ 10xxxxx│
  └──────────┴──────┴──────┴──────┴──────┴────────┘
       Td: architectural tile register (T0–T31) — read-modify-write
       Rs: scalar GPR holding element value
       Ridx: scalar GPR holding element index
       funct3: element type
```

Every MTE instruction flows through both the **Scalar RAT** (for address/data operands) and the **Tile RAT** (for tile operands) at the D2 rename stage:

| Instruction | Scalar RAT | Tile RAT source(s) | Tile RAT destination | Result bus |
|-------------|-----------|---------------------|----------------------|------------|
| TILE.LD Td, [Rbase] | Rbase → P-reg lookup | — | Td → allocate new PT | TCB |
| TILE.LD Td, [Rbase], Rs | Rbase, Rs → P-reg lookups | — | Td → allocate new PT | TCB |
| TILE.ST [Rbase], Ts | Rbase → P-reg lookup | Ts → PT lookup | — | — |
| TILE.ST [Rbase], Ts, Rs | Rbase, Rs → P-reg lookups | Ts → PT lookup | — | — |
| TILE.GATHER Td, [Rbase], Tidx | Rbase → P-reg lookup | Tidx → PT lookup | Td → allocate new PT | TCB |
| TILE.SCATTER [Rbase], Ts, Tidx | Rbase → P-reg lookup | Ts, Tidx → PT lookups | — | — |
| TILE.ZERO Td | — | — | Td → allocate new PT | TCB |
| TILE.COPY Td, Ts | — | Ts → PT lookup | Td → allocate new PT | TCB |
| **TILE.MOVE Td, Ts** | — | Ts → PT lookup | **Td → alias PT(Ts)** (no alloc) | **— (rename-only)** |
| **TILE.TRANSPOSE Td, Ts, fmt** | — | Ts → PT lookup | Td → allocate new PT | TCB |
| **TILE.GET Rd, Ts, Ridx** | Ridx → P-reg lookup; **Rd → allocate new P-reg** | Ts → PT lookup | — | **CDB** (scalar) |
| **TILE.PUT Td, Rs, Ridx** | Rs, Ridx → P-reg lookups | **Td → PT lookup (old)** | **Td → allocate new PT** | TCB |

Key observations:
- **TILE.MOVE** is handled entirely at D2 rename (**move elimination**): Tile RAT[Td] is pointed to the same physical tile as Ts. No free-list allocation, no RS entry, no execute stage, no result bus. Zero-cycle latency.
- **TILE.TRANSPOSE** allocates a new physical tile and requires a full read-then-transpose-then-write pass through the MTE's transpose buffer.
- **TILE.GET** produces a **scalar GPR result** (broadcast on CDB), while consuming a tile source. It requires both a Tile RAT source lookup and a Scalar RAT destination allocation.
- **TILE.PUT** is a **read-modify-write** on the tile: the rename stage looks up the old physical tile mapping as a source AND allocates a new physical tile as a destination. The MTE unit copies the old tile contents to the new tile, then overwrites the single element.

After rename, MTE RS entries carry physical scalar register tags (from Scalar RAT) and physical tile tags (from Tile RAT). The MTE unit maintains a large outstanding request buffer to maximize memory-level parallelism.

#### 2.4.4 v2 实现层增量(对软件不可见)

1. **`TILE.TRANSPOSE` becomes a software-optional accelerator.** With per-port `is_transpose` on the TRegFile read (§9.2) and per-beat `tilelet_xpose` in the vector unit (§8.3), most "pre-transpose then consume" patterns become single-instruction with `is_xpose_*` set on the consuming op. `TILE.TRANSPOSE` is retained for cases that need a *materialized* transposed tile reused many times across instructions that themselves don't carry the bit; its physical staging buffer shrinks from 4 KB → 512 B (§8.5.1).
2. **All bulk tile stores (`TILE.ST`, `TILE.SCATTER`) acquire a branch tag at dispatch** and are gated through the **Speculative Tile-Store Queue** (STQ, §11.5) until their tag becomes non-speculative. Invisible at the ISA level; adds 0–6 cycles of latency to a tile store on the speculative path.

### 2.5 Instruction Domain Identification

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §2.5。)**

The 7-bit opcode field encodes the instruction domain:

| Opcode[6:5] | Domain | Decode path |
|-------------|--------|-------------|
| 00, 01 | Scalar | Scalar rename → Scalar RS |
| 10 | Vector / MTE | Tile RAT rename → Vector RS or MTE RS |
| 11 | Cube | Tile RAT rename → Cube RS |

---

## 3. Top-Level Block Diagram

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  DAVINCI-v2 CORE                                                                     │
 │                                                                                      │
 │  ┌─────────────────────────────── FRONT-END ──────────────────────────────────────┐  │
 │  │   ┌──────────┐    ┌───────────┐    ┌──────────────────────────────────┐        │  │
 │  │   │  Branch   │───▶│  Fetch    │───▶│  Instruction Buffer (16 entries) │        │  │
 │  │   │ Predictor │    │  Unit     │    │  4-wide dequeue                  │        │  │
 │  │   │ TAGE+BTB  │    │ (L1-I)    │    └──────────┬───────────────────────┘        │  │
 │  │   │ +RAS      │    └──────────┘               │ 4 instr/cy + branch tag         │  │
 │  │   └──────────┘                                ▼                                │  │
 │  │                              ┌───────────────────────────────────────────┐     │  │
 │  │                              │  Decode + Rename (4-wide)                  │     │  │
 │  │                              │  ┌───────┐  ┌──────────────┐               │     │  │
 │  │                              │  │Scalar │  │  Tile RAT     │               │     │  │
 │  │                              │  │ RAT   │  │  (32→256)     │               │     │  │
 │  │                              │  └───┬───┘  └──────┬───────┘               │     │  │
 │  │                              │  ┌───┴─────────────┴────────┐              │     │  │
 │  │                              │  │ Free Lists + Ref Counters  │              │     │  │
 │  │                              │  └───────────────────────────┘              │     │  │
 │  │                              │  ┌───────────────────────────┐              │     │  │
 │  │                              │  │ Checkpoint Store (8 slots) │              │     │  │
 │  │                              │  │ + Branch-tag allocator    │              │     │  │
 │  │                              │  └───────────────────────────┘              │     │  │
 │  │                              │  ┌───────────────────────────┐              │     │  │
 │  │                              │  │ Tile Metadata RAT (32→256)│              │     │  │
 │  │                              │  │  (32 b per phys tile)     │              │     │  │
 │  │                              │  └───────────────────────────┘              │     │  │
 │  │                              └──────────────┬──────────────────────────────┘     │  │
 │  └─────────────────────────────────────────────┼─────────────────────────────────────┘  │
 │                                                │ renamed µops + branch_tag              │
 │  ┌─────────────────────────── DISPATCH ────────┼───────────────────────────────────┐  │
 │  │                                             ▼                                  │  │
 │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
 │  │  │Scalar RS │  │  LSU RS  │  │Vector RS │  │ Cube RS  │  │  MTE RS  │         │  │
 │  │  │(32 entry)│  │(24 entry)│  │(24 entry)│  │(4 entry) │  │(16 entry)│         │  │
 │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │  │
 │  └───────┼──────────────┼─────────────┼─────────────┼──────────────┼───────────────┘  │
 │          │              │             │             │              │                  │
 │  ┌───────┼──────────── EXECUTE ───────┼─────────────┼──────────────┼───────────────┐  │
 │  │       ▼              ▼             ▼             ▼              ▼              │  │
 │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
 │  │  │ 4× ALU   │  │  Load /  │  │  VEC-    │  │outerCube │  │   MTE    │         │  │
 │  │  │ 1× MUL   │  │  Store   │  │  4K-v2   │  │   MXU    │  │  Engine  │         │  │
 │  │  │ 1× BRU   │  │  Unit    │  │ 3R/2W    │  │(4096 MAC)│  │(LD/ST/  │         │  │
 │  │  │          │  │  + SSB   │  │ tiles    │  │          │  │G/S/MOVE)│         │  │
 │  │  │          │  │  (24)    │  │ 1 tile/  │  │          │  │+ STQ(8) │         │  │
 │  │  │          │  │          │  │ 8 cy     │  │          │  │         │         │  │
 │  │  └─────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘         │  │
 │  └────────┼─────────────┼─────────────┼─────────────┼──────────────┼─────────────┘  │
 │           │             │             │             │              │                  │
 │  ┌────────┼──────────── COMPLETE ─────┼─────────────┼──────────────┼───────────────┐  │
 │  │  CDB (6 ports, scalar)  +  TCB (4 ports, tile)                                 │  │
 │  └────────────────────────────────────────────────────────────────────────────────┘  │
 │                                                                                      │
 │  ┌──────────────── REGISTER FILES ───────────────────────────────────────────────┐  │
 │  │  ┌──────────────────┐  ┌─────────────────────────────────────────────────────┐ │  │
 │  │  │ Scalar Physical   │  │ TRegFile-4K (with per-port is_transpose)            │ │  │
 │  │  │ Register File     │  │ 256×4KB = 1MB; 8R+8W @ 512B/cy/port                  │ │  │
 │  │  │ 128×64b           │  │ 8-cycle epoch calendar                                │ │  │
 │  │  │ 12R+6W ports      │  │ 32 b metadata per physical tile                       │ │  │
 │  │  │                   │  │ row-mode AND col-mode reads at full 512 B/cy          │ │  │
 │  │  └──────────────────┘  └─────────────────────────────────────────────────────┘ │  │
 │  └──────────────────────────────────────────────────────────────────────────────────┘  │
 │                                                                                      │
 │  ┌──────────────── MEMORY SUBSYSTEM ─────────────────────────────────────────────┐  │
 │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                                    │  │
 │  │  │  L1-I    │  │  L1-D    │  │  L2 (512 │───▶ External Bus / NoC              │  │
 │  │  │  64 KB   │  │  64 KB   │  │   KB)    │                                    │  │
 │  │  └──────────┘  └────┬─────┘  └──────────┘                                    │  │
 │  │                     ▲                                                        │  │
 │  │            ┌────────┴────────┐                                                │  │
 │  │            │ Spec Store Buf  │  (24 entries, branch-tag gated, §11.4)        │  │
 │  │            │ (SSB)           │                                                │  │
 │  │            └─────────────────┘                                                │  │
 │  │                     ▲                                                        │  │
 │  │            ┌────────┴────────┐                                                │  │
 │  │            │ Spec Tile-Store │  (8 entries, branch-tag gated, §11.5)         │  │
 │  │            │ Queue (STQ)     │  drains TILE.ST / TILE.SCATTER                 │  │
 │  │            └─────────────────┘                                                │  │
 │  │                     ▲                                                        │  │
 │  │                     │ from MTE                                               │  │
 │  └──────────────────────────────────────────────────────────────────────────────┘  │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

**v2 deltas highlighted:**

- Branch-tag allocator at rename / checkpoint store.
- Tile Metadata RAT (32 b per physical tile) co-located with the Tile RAT.
- VEC-4K-v2 unit replaces v1 vector unit; 3R / 2W tile interface.
- Speculative Store Buffer (SSB, 24 entries) gates scalar stores by branch tag.
- Speculative Tile-Store Queue (STQ, 8 entries) gates MTE bulk stores by branch tag.

---

## 4. Pipeline Overview

> **(v1 → v2: 4.A / 4.B / 4.C 完整复制自 v1 §4.1 / §4.2 / §4.3。流水线深度、阶段定义、各域执行延迟未变更。v2 增量集中在 §4.1 / §4.2 / §4.3。)**

The v2 scalar pipeline is the same **12 stages** as v1 (no retire/commit stage). Branch-tag administration adds zero cycles — tags are allocated at D2 (alongside the existing checkpoint allocation) and propagated forward as one extra metadata field per RS entry.

```
 F1 → F2 → D1 → D2 → DS → IS → EX1 → EX2 → EX3 → EX4 → WB → (no retire)
 ├── Fetch ──┤├─ Decode/Rename ─┤├DS┤├IS┤├──── Execute ────────┤├WB┤
                       │
                       └── allocate branch_tag ∈ {0..7} for each in-flight branch
                       └── flash-copy {Scalar RAT, Tile RAT, Tile-Meta RAT,
                                       free-list heads, RAS top, SSB head, STQ head}
                                       into checkpoint[tag]
```

### 4.A Pipeline Stages (v1 §4.1, 未变更)

| Stage | Name | Function |
|-------|------|----------|
| F1 | Fetch-1 | Send PC to L1-I cache and branch predictor |
| F2 | Fetch-2 | Receive 4 instructions from I-cache; apply BTB/TAGE prediction |
| D1 | Decode | Decode 4 instructions; identify domain (scalar/vector/cube/MTE) |
| D2 | Rename | Read Scalar RAT + Tile RAT (+ Tile-Meta RAT in v2); allocate physical registers/tiles; intra-group bypass; checkpoint all RATs on branch; *(v2)* allocate branch_tag |
| DS | Dispatch | Allocate reservation station entry; write operand tags/data; *(v2)* allocate SSB/STQ slot for memory-side stores |
| IS | Issue | Select oldest ready instruction per functional unit; read physical RF |
| EX1–EXn | Execute | Variable latency: ALU=1cy, MUL=4cy, LD=4cy(L1 hit), VEC=16cy(2 epochs, elementwise), MTE=8–72cy(epoch+mem), DIV=12–20cy |
| WB | Writeback | Broadcast result on CDB/TCB; write to physical RF / TRegFile-4K; wakeup dependent RS entries; *(v2)* SSB/STQ entries become drain-ready when branch_tag clears |

### 4.B Pipeline Timing — Scalar ALU Instruction (v1 §4.2, 未变更)

```
  Cycle:  0    1    2    3    4    5    6    7
  ─────  ────  ────  ────  ────  ────  ────  ────
  i0:    F1   F2   D1   D2   DS   IS   EX1  WB
  i1:    F1   F2   D1   D2   DS   IS   EX1  WB
  i2:    F1   F2   D1   D2   DS   IS   EX1  WB
  i3:    F1   F2   D1   D2   DS   IS   EX1  WB
         └──── 4-wide ────────┘
```

### 4.C Execution Latencies by Domain (v1 §4.3, 未变更)

| Domain | Operation | Stages | Latency (cycles) | Pipelined |
|--------|-----------|--------|-------------------|-----------|
| Scalar | ALU (add, logic, shift) | EX1 | **1** | yes |
| Scalar | MUL | EX1–EX4 | **4** | yes |
| Scalar | DIV | EX1–EX(12–20) | **12–20** | no |
| Scalar | Branch resolve | EX1 | **1** | yes |
| LSU | Load (L1 hit) | EX1–EX4 | **4** | yes |
| LSU | Load (L2 hit) | EX1–EX(12) | **12** | yes |
| LSU | Store | EX1–EX4 | **4** (addr+data) | yes |
| Vector | VADD/VMUL/VFMA (full tile, elementwise) | 2 epochs (16 cy) | **16** (8 read + 8 write, compute hidden) | epoch-pipelined |
| Vector | Reduce (VROWSUM/VCOLSUM/...) | 1 epoch + reduce | **16** (8 read + reduce + 8 write) | no |
| Vector (v2) | TINV (128×128 FP32) | multi-epoch | **~33 K beats** (~22 µs @ 1.5 GHz) | no |
| Vector (v2) | TMRGSORT (1024 FP32) | multi-epoch | **~220 cy** | no |
| Cube | CUBE.OPA (N steps) | 19 + N | **N + 18** (first tile) | epoch-pipelined |
| MTE | TILE.LD (contiguous, L2 hit) | mem + 1 write epoch | **72** (64 mem + 8 TRegFile write) | yes (across ports) |
| MTE | TILE.ST (contiguous, L2) | 1 read epoch + mem | **72** (8 TRegFile read + 64 mem write) | yes (across ports) |
| MTE | TILE.COPY | 2 epochs | **16** (8 read + 8 write) | epoch-pipelined |
| MTE | TILE.ZERO | 1 write epoch | **8** (write zeros, no read) | yes |
| MTE | TILE.GATHER (L2 hit) | mem + 1 write epoch | **72–128** (variable mem + 8 TRegFile write) | partially |
| MTE | TILE.SCATTER (L2) | 1 read epoch + mem | **72–128** (8 TRegFile read + variable mem) | partially |
| MTE | TILE.MOVE (rename-only) | — (D2) | **0** (move elimination, no execute) | — |
| MTE | TILE.TRANSPOSE | 2 epochs | **16** (8 read + 8 write via transpose buffer) | no |
| MTE | TILE.GET (element → GPR) | 1 read epoch + extract | **9** (8 TRegFile read epoch + 1 extract) | no (port occupied 8 cy) |
| MTE | TILE.PUT (GPR → element, RMW) | 2 epochs | **16** (8 read + 8 write), **8** with copy elision | no |

---

### 4.1 Per-stage actions (v2 deltas)

| Stage | v2 deltas |
|-------|-----------|
| **D2 — Rename** | Allocate branch_tag if instruction is a branch; tag propagates to every younger instruction's RS entry. Checkpoint (§5/§6) now also snapshots **SSB head pointer** and **STQ head pointer**. |
| **DS — Dispatch** | Dispatched RS entry includes `branch_tag` (3 b). LSU stores additionally allocate an SSB slot tagged with the same branch_tag. MTE bulk-stores allocate an STQ slot. |
| **IS — Issue** | No change (RS wakeup on tag-match for source operands). |
| **EX — Execute** | LSU stores deposit data + address into the SSB slot, but do **not** commit to L1-D. MTE bulk-stores deposit address-list + data into the STQ slot, but do **not** drain to memory. Both wait for `branch_tag → non-speculative` (§11.4, §11.5). |
| **WB — Writeback** | Unchanged for scalar/tile *register* destinations. Memory-side commits gated on tag-clear from the speculation tracker. |

### 4.2 Pipeline timing (v2 增量说明)

Scalar ALU, MUL, LD, branch resolve, and elementwise-vector timing (epoch-pipelined at 1 tile/8 cy) are all preserved as in §4.B / §4.C above. The only timing change in v2 is the **variable operand-fetch prologue** of VEC-4K-v2 (§8.3.6): `T_fetch ∈ {8 cy, 16 cy}` depending on `is_xpose_*` mix among the 3 source tiles. End-to-end vector-instruction latency is unchanged for uniformly-transposed (or uniformly-non-transposed) operands.

### 4.3 Branch misprediction penalty

| Step | v1 | v2 |
|------|----|----|
| Detection (EX1) | 1 cy | 1 cy |
| Recovery (RAT flash-restore + flush + redirect) | 1 cy | 1 cy |
| **+ SSB / STQ flush** | — | concurrent with RAT restore (§11.4.4) |
| Front-end refill | ~6 cy | ~6 cy |
| **Total mispredict penalty** | **6 cy** | **6 cy** (unchanged) |

The SSB / STQ flush runs in parallel with the RAT flash-restore: both are mask-clear operations on small CAMs, single-cycle.

---

## 5. Front-End: Fetch & Branch Prediction

> **(v1 → v2: 子节 5.A / 5.B / 5.C 完整复制自 v1 §5.1 / §5.2 / §5.3,内容未变更。v2 增量为 §5.1 Branch-tag allocator 与 §5.2 Static hint bit。)**

### 5.A Fetch Unit (v1 §5.1, 未变更)

The fetch unit delivers up to **4 aligned instructions per cycle** from the L1 instruction cache.

| Parameter | Value |
|-----------|-------|
| Fetch width | **4** instructions / cycle (16 bytes) |
| Fetch alignment | 16-byte aligned fetch block |
| Instruction buffer | **16** entries (4-cycle decoupling) |
| L1-I cache | **64 KB**, 4-way set-associative, 64 B line |
| L1-I latency | **2** cycles (F1 + F2) |
| I-TLB | 64 entries, fully associative |

**Fetch pipeline:**

```
  F1: PC → I-TLB + L1-I tag lookup + BTB lookup + TAGE index
  F2: L1-I data return (4 instructions) + TAGE prediction + RAS check
      → push into instruction buffer (up to 16 entries)
      → if predicted-taken: redirect PC at end of F2
```

### 5.B Branch Predictor (v1 §5.2, 未变更)

The branch predictor uses a **hybrid scheme** combining three components.

#### 5.B.1 TAGE Predictor (Conditional Branches)

| Parameter | Value |
|-----------|-------|
| Base predictor | 4K-entry bimodal (2-bit saturating counters) |
| Tagged tables | 5 tables: T1(512), T2(512), T3(1K), T4(1K), T5(1K) |
| History lengths | 4, 8, 16, 32, 64 (geometric series) |
| Tag width | 8–12 bits per entry |
| Total storage | ~20 KB |
| Prediction accuracy | ~95% (typical workloads) |

#### 5.B.2 Branch Target Buffer (BTB)

| Parameter | Value |
|-----------|-------|
| Entries | **2048** |
| Associativity | 4-way set-associative |
| Tag | partial PC (upper bits) |
| Target | full 64-bit target address |
| Hit latency | 1 cycle (available end of F1) |

#### 5.B.3 Return Address Stack (RAS)

| Parameter | Value |
|-----------|-------|
| Depth | **16** entries |
| Push | on JAL/JALR to link register |
| Pop | on JALR from link register (return pattern) |
| Speculative management | checkpoint RAS top-of-stack pointer with RAT checkpoints |

### 5.C Fetch Redirect Priorities (v1 §5.3, 未变更)

```
  Priority (highest to lowest):
    1. Branch mispredict redirect (from EX1)  — flush + restart
    2. BTB/TAGE taken-branch redirect (from F2) — next-cycle redirect
    3. Sequential PC+16 (default)
```

---

### 5.1 Branch-tag allocator (v2 增量)

A small hardware counter at D2 allocates a 3-bit branch_tag for each newly-decoded branch, drawn from the same 8-slot pool used by the v1 RAT-checkpoint store. Allocation policy is round-robin among free slots; when the pool is empty, the rename stage stalls (same condition as v1's checkpoint-pool exhaustion).

The branch_tag is then attached to:

- The branch's own RS entry.
- All RS entries dispatched **after** the branch and **before** any older branch resolves.
- All SSB entries created in the same window.
- All STQ entries created in the same window.
- Free-list pointers in the checkpoint snapshot.

When the branch resolves correctly, the tag is freed and propagated as a "tag-clear" event to all consumers (RS / SSB / STQ). When it mispredicts, the tag becomes the "flush key" — every entry tagged with this branch (or any *younger* branch tag) is invalidated atomically (§11.4.4).

### 5.2 Static hint bit (v2 增量)

The compiler may set the conditional-branch funct3's `H` bit (1 = predict taken on TAGE/BTB miss). The hint is consulted only on a predictor cold-miss; once TAGE has trained on the branch, dynamic prediction wins.

---

## 6. Decode & Rename

> **(v1 → v2: 子节 6.A / 6.B 完整复制自 v1 §6.1 / §6.2,内容未变更。v2 增量为 §6.1 Tile Metadata RAT、§6.2 Branch-tag stamping、§6.3 Checkpoint extensions。)**

### 6.A Decode Stage (D1) — (v1 §6.1, 未变更)

The decode stage processes **4 instructions per cycle**, identifying each instruction's domain, opcode, source/destination registers, and immediate values.

| Function | Detail |
|----------|--------|
| Decode width | **4** instructions / cycle |
| Domain classification | Opcode[6:5] → scalar, vector, cube, MTE |
| Immediate extraction | Sign-extend and format-dependent extraction |
| Branch detection | Identify branch instructions for checkpoint allocation |

Instructions that cannot be expressed as a single micro-op (e.g., certain complex addressing modes) are **cracked** into 2 micro-ops at D1, consuming 2 dispatch slots.

### 6.B Rename Stage (D2) — (v1 §6.2, 未变更)

The rename stage performs **register renaming** for both scalar registers and tile registers using two independent Register Alias Tables (RATs). The Scalar RAT maps 32 architectural GPRs to 128 physical GPRs. The Tile RAT maps 32 architectural tile registers to 256 physical tile slots in TRegFile-4K.

#### 6.B.1 Scalar RAT (v1 §6.2.1)

| Parameter | Value |
|-----------|-------|
| Architectural registers | 32 (X0–X31) |
| Physical registers | **128** (P0–P127) |
| RAT storage | 32 entries × 7 bits = **224 bits** |
| Read ports | **8** (2 sources × 4 decode slots) |
| Write ports | **4** (1 destination × 4 decode slots) |

Each RAT entry contains:
- Physical register index (7 bits)
- Ready bit (1 bit): set when result has been written to the physical RF

#### 6.B.2 Tile RAT (Vector, Cube, MTE Operands) — (v1 §6.2.2)

All three tile-consuming domains (vector, cube, MTE) share a single **Tile RAT** that renames 32 architectural tile registers (T0–T31) to 256 physical tile slots (PT0–PT255) in TRegFile-4K. This eliminates WAW and WAR hazards on tile operands through renaming, exactly as the scalar RAT does for GPRs.

| Parameter | Value |
|-----------|-------|
| Architectural tile registers | 32 (T0–T31) |
| Physical tile registers | **256** (PT0–PT255), 4 KB each in TRegFile-4K |
| Tile RAT storage | 32 entries × 8 bits = **256 bits** |
| Read ports | **8** (up to 3 source tiles × 4 decode slots, shared/muxed) |
| Write ports | **4** (1 destination tile × 4 decode slots) |

Each Tile RAT entry contains:
- Physical tile index (8 bits)
- Ready bit (1 bit): set when the producing operation has finished writing the physical tile

Tile RAT operation mirrors scalar RAT operation: at D2, destination tile operands are allocated a fresh physical tile from the tile free list, the old physical tile mapping is marked orphan, and source tile operands are looked up to obtain the current physical tile index and ready bit. Tile instructions dispatched to Vector RS, Cube RS, and MTE RS carry **physical tile tags** (8 bits each) rather than architectural indices.

#### 6.B.3 Intra-Group Bypass Logic (v1 §6.2.3)

When 4 instructions are renamed simultaneously, later instructions in the group may depend on earlier ones. Hardware **priority-encoded comparators** detect these intra-group dependencies for both scalar and tile RATs:

```
  Scalar example:
    Rename slot 0:  X5 → P40  (destination)
    Rename slot 1:  reads X5  → comparator detects match → bypass P40
    Rename slot 2:  X5 → P41  (re-definition)
    Rename slot 3:  reads X5  → comparator detects slot 2 match → bypass P41

    4 slots × 2 sources × 3 older slots = 24 comparators (7-bit each)
    + 8 bypass MUXes (select forwarded phys-reg vs scalar RAT read)

  Tile example:
    Rename slot 0:  TILE.LD T10  → PT200  (destination)
    Rename slot 1:  VADD dst=T10 → PT201  (re-definition)
    Rename slot 2:  reads T10    → comparator detects slot 1 match → bypass PT201

    4 slots × 3 tile sources × 3 older slots = 36 comparators (8-bit each)
    + 12 bypass MUXes (select forwarded phys-tile vs Tile RAT read)
```

#### 6.B.4 Free Lists (v1 §6.2.4)

| Parameter | Value |
|-----------|-------|
| **Scalar free list** | FIFO, 96 entries (128 physical − 32 architectural) |
| Scalar dequeue rate | up to 4 per cycle |
| Scalar enqueue rate | up to 4 per cycle (from ref-count freeing) |
| **Tile free list** | FIFO, 224 entries (256 physical − 32 architectural) |
| Tile dequeue rate | up to 4 per cycle |
| Tile enqueue rate | up to 4 per cycle (from tile ref-count freeing) |

At reset, the scalar free list is initialized with P32–P127 (first 32 pre-assigned to X0–X31). The tile free list is initialized with PT32–PT255 (first 32 pre-assigned to T0–T31).

**Stall condition:** If either free list cannot supply enough physical registers for the current decode group, the rename stage stalls the pipeline.

#### 6.B.5 Checkpoint Storage (Branch Recovery) — (v1 §6.2.5, 在 v2 §6.3 中扩展)

| Parameter | Value (v1 baseline) |
|-----------|---------------------|
| Checkpoint slots | **8** (supports 8 in-flight unresolved branches) |
| Checkpoint size (v1) | Scalar RAT (224b) + Tile RAT (256b) + scalar free-list head (7b) + tile free-list head (8b) + RAS pointer (4b) = **~499 bits** |
| Flash-copy latency | **1 cycle** (parallel bit-copy from both active RATs) |
| Flash-restore latency | **1 cycle** (parallel bit-copy to both active RATs) |

Both the Scalar RAT and Tile RAT are checkpointed. On branch misprediction, both RATs are restored in parallel, along with both free-list head pointers.

**Checkpoint lifecycle (v1 baseline):**

```
  Branch decoded at D2:
    1. Allocate checkpoint slot (round-robin)
    2. Flash-copy: active Scalar RAT + active Tile RAT → checkpoint[i]
    3. Save scalar free-list head, tile free-list head, and RAS top pointer
    4. Tag the branch's RS entry with checkpoint ID

  Branch resolved correctly at EX1:
    1. Deallocate checkpoint slot → available for reuse

  Branch mispredicted at EX1:
    1. Flash-restore: checkpoint[i] → active Scalar RAT + active Tile RAT
    2. Restore scalar free-list head pointer (reclaim speculatively allocated GPRs)
    3. Restore tile free-list head pointer (reclaim speculatively allocated tiles)
    4. Restore RAS pointer
    5. Flush all pipeline stages after D2
    6. Redirect fetch to correct target
```

**Stall condition:** If all 8 checkpoint slots are occupied, the rename stage stalls on the next branch instruction until an older branch resolves.

---

### 6.1 Tile Metadata RAT (v2 增量)

A new **32 b × 256 entry** SRAM stores the metadata word for each physical tile. Access pattern:

| Event | Action |
|-------|--------|
| Physical tile allocated (D2 of a producer instruction) | Reset metadata to "uninitialized" sentinel |
| Producer retires | Producer writes metadata word (alongside its 4 KB payload) |
| Consumer reads tile | Consumer reads metadata word from the **first strip**'s arrival at staging (§8.3.3) |

The Tile Metadata RAT is read at D2 alongside the regular Tile RAT lookup; the metadata word is forwarded to the consumer's VEC RS entry (`SOP` staging at issue, [`vector4k_v2.md`](vector4k_v2.md) §4.4).

`TSETMETA` writes the metadata RAT directly at D2, with no execute stage.

**Storage:** 256 × 32 b = 1024 B = ~10 K gate. Read ports: 4 (one per decode slot) + 1 (TCB completion). Write ports: 2 (1 at retire, 1 for `TSETMETA`).

### 6.2 Branch-tag stamping (v2 增量)

Every µop entering the rename stage receives the **current** branch_tag (the tag of the youngest unresolved branch ahead of it; or `0xFF` if none). Storing this tag on the µop's RS entry (3 b) lets the misprediction recovery logic identify which entries to flush in one cycle.

### 6.3 Checkpoint extensions (v2 增量)

The 8 RAT-checkpoint slots are extended to also snapshot:

- SSB head pointer (5 b).
- STQ head pointer (4 b).
- Tile Metadata RAT delta-vector (4 b "tags last touched" — the metadata RAT itself doesn't need full snapshot because metadata writes are tied to retire, not rename; only the *delta-tag* needs replay).

Updated checkpoint size:

| Field | v1 | v2 |
|-------|----|----|
| Scalar RAT | 224 b | 224 b |
| Tile RAT | 256 b | 256 b |
| Scalar free-list head | 7 b | 7 b |
| Tile free-list head | 8 b | 8 b |
| RAS top pointer | 4 b | 4 b |
| **SSB head pointer** | — | 5 b |
| **STQ head pointer** | — | 4 b |
| **Metadata-delta tag** | — | 4 b |
| **Per-checkpoint total** | ~499 b | **~512 b** |

8 slots × 512 b = **4 KB checkpoint store** (negligible vs. the 1 MB TRegFile). Flash-restore latency stays at **1 cycle**.

---

## 7. Dispatch & Issue

### 7.1 Dispatch (DS)

> **(v1 → v2: §7.1 整体语义未变,仅 Vector RS 容量从 16 → 24,理由见下表。其余 RS 容量与分发流程完整继承自 v1 §7.1。)**

After rename, each µop dispatches into the appropriate RS based on its domain and operation type. Dispatch is **in-order** (preserving program order for dependency tracking), but issue from reservation stations is **out-of-order**. Each µop carries its `branch_tag` (3 b) into the RS entry alongside the v1 fields.

| Reservation Station | Serves | v1 entries | v2 entries | Issue width | Sizing rationale |
|---------------------|--------|------------|------------|-------------|------------------|
| **Scalar RS** | 4× ALU, 1× MUL/DIV, 1× BRU | **32** | **32** | 6 (4 ALU + 1 MUL + 1 BRU) | unchanged |
| **LSU RS** | Load unit, Store unit | **24** | **24** | 2 (1 load + 1 store) | unchanged |
| **Vector RS** | VEC-4K-v2 ALU/FMA/PTO | **16** | **24** | 1 | VEC-4K-v2 entries are wider (3 source tile tags + 2 dest + mask flag + xpose flags + retire mask + scalar staging tags) and multi-cycle ops occupy slots longer (`TINV` up to 33 K beats; `TMRGSORT` up to 2 912 beats — fewer in number than elementwise ops, but their long execute time means the RS must absorb more dispatched ops without stalling the front-end). |
| **Cube RS** | CUBE.OPA, CUBE.CFG, CUBE.DRAIN, CUBE.ZERO, CUBE.WAIT | **4** | **4** | 1 | unchanged |
| **MTE RS** | TILE.LD/ST/GATHER/SCATTER/GET/PUT/COPY/TRANSPOSE | **16** | **16** | 2 | unchanged |
| **Total** | — | **92** | **100** | — | +8 vector RS entries |

#### 7.1.1 Reservation Station Sizing Rationale (v1 §7.1.1, 未变更 except Vector RS as noted)

> **(v1 → v2: 以下 5 个子段中 4 个完整复制自 v1 §7.1.1。Vector RS 子段更新以反映 v2 容量提升至 24。)**

The number of entries in each RS is chosen to satisfy: **(1)** absorb the execution latency of its functional units so that new instructions are not stalled waiting for RS slots, **(2)** provide enough window for out-of-order issue to find independent instructions, and **(3)** stay within area and wakeup-logic power budgets. The core principle: RS entries ≈ **dispatch rate × average occupancy time**, with headroom for dependent chains and dispatch bursts.

**Scalar RS — 32 entries (v1, 未变更):**
- The front-end dispatches up to **4 instructions/cycle**. In typical AI/HPC kernels, ~60–70% of instructions are scalar (address computation, loop control, branch), giving ~2.5–3 scalar dispatches/cycle.
- ALU latency is **1 cycle** (result available next cycle), so independent ALU chains drain quickly. However, **MUL (4 cy)** and **DIV (12–20 cy)** are multi-cycle and block their pipeline slot while in-flight. A single DIV can occupy an issue port for up to 20 cycles.
- With 6 issue ports, up to 6 instructions leave the RS per cycle, but dependent chains create bubbles. The RS needs enough depth to look past these stalls and find independent operations.
- Sizing: ~3 dispatch/cy × ~8 cy average occupancy (mix of 1-cy ALU and 4-cy MUL, with occasional 20-cy DIV) ≈ 24 entries minimum. Rounded up to **32** to tolerate bursty dispatch and long DIV chains.
- Wakeup cost: 32 entries × 2 sources × 6 CDB ports = **384 tag comparators** (7-bit each) — acceptable at 5 nm.

**LSU RS — 24 entries (v1, 未变更):**
- Load/store instructions make up ~20–30% of a typical mix, so ~0.8–1.2 dispatches/cycle.
- **Load latency is 4 cycles** (L1 hit) but **12+ cycles** on L1 miss (L2 hit), and hundreds of cycles on DRAM access. The LSU RS must buffer many outstanding loads to exploit **memory-level parallelism (MLP)**.
- The L1-D cache supports **8 MSHRs** (miss-status holding registers) — up to 8 cache misses can be in flight simultaneously, each occupying an RS slot for 12+ cycles.
- Sizing: 8 MSHR-bound loads (occupying slots for ~12 cy each) + steady-state L1-hit loads and stores ≈ **24 entries**. This keeps the memory subsystem saturated with overlapping misses while allowing hit-path traffic to proceed without stalling dispatch.
- The 2-port issue (1 load + 1 store/cycle) prevents store traffic from blocking load throughput.
- **(v2 增量, §11.4)**: Each LSU RS entry now carries an additional 5-bit `ssb` field pointing into the Speculative Store Buffer for stores; pure loads leave it unused.

**Vector RS — 24 entries (v2 update from v1's 16):**
- Vector instructions have **high latency** (16 cycles for elementwise; up to 33 K beats for `TINV`; up to 2 912 beats for `TMRGSORT`) with epoch-pipelined throughput of **1 tile per 8 cycles** for elementwise ops. They are less frequent than scalar ops but arrive in bursts during vector-heavy code regions.
- v2 RS entries are **wider** than v1: 3 source tile tags (was 2) + 2 destination tile tags (was 1) + mask-flag + 3 per-operand xpose bits + retire-mask + scalar staging tags. Each tile-domain entry is ~92 b (was ~80 b).
- v2 also adds long multi-cycle PTO ops (`TINV`, `TROWRANGE_MUL`, `TMRGSORT`) that can occupy an RS slot for thousands of beats; the RS must absorb additional dispatched ops without stalling the front-end during these long ops.
- Sizing: v1's 16 entries → v2's **24 entries** (+8 entries, +50%).
- Tile-domain RS entries don't capture 4 KB tile data: only 8-bit tags + ready bits, so 24 entries ≈ **276 bytes**.

**Cube RS — 4 entries (v1, 未变更):**
- Cube instructions are **very long-latency** (CUBE.OPA: N+18 cycles, typically 26–82 cy) but **extremely infrequent** — a single CUBE.OPA encodes an entire K-loop of outer-product-accumulate steps spanning thousands of MAC operations.
- A typical GEMM kernel issues 1 CUBE.OPA per ~100+ scalar/MTE instructions. Software double-buffers tile loads around cube execution, so the cube RS is rarely the dispatch bottleneck.
- The RS only needs to hold: the currently-executing CUBE.OPA, the next queued CUBE.OPA (overlapping with tile loads for the next iteration), plus associated CUBE.CFG/CUBE.DRAIN/CUBE.WAIT control instructions.
- Sizing: **4 entries** suffices because the instruction stream rarely has >2–3 cube instructions queued. Additional entries would waste area (including 8-bit tile tag comparators) with no throughput benefit since the cube pipeline executes 1 instruction at a time.

**MTE RS — 16 entries (v1, 未变更):**
- MTE instructions span a wide latency range: **TILE.LD: 72 cy** (L2 hit, potentially hundreds from DRAM), **TILE.ST: 72 cy**, **TILE.COPY/TRANSPOSE: 16 cy**, **TILE.ZERO: 8 cy**, **TILE.GET: 9 cy**.
- The primary design driver is **memory-level parallelism for tile loads**: the programmer (or compiler) schedules many TILE.LD instructions ahead of the CUBE.OPA that consumes the loaded tiles. With up to **7 available write ports** and a **32-entry outstanding request buffer**, the MTE can service many concurrent tile loads.
- At 2 issues/cycle, the MTE RS can launch 2 tile operations per cycle (e.g., 1 TILE.LD + 1 TILE.ST on separate ports).
- Sizing: 7 concurrent TILE.LDs (one per write port) + several TILE.STs and local tile ops (GET/PUT/COPY/TRANSPOSE) + headroom for dispatch bursts ≈ **16 entries**. This provides enough scheduling window to overlap tile loads with stores and local operations, maximizing TRegFile-4K port utilization.
- **(v2 增量, §11.5)**: Each MTE RS entry for `TILE.ST`/`TILE.SCATTER` now carries an additional 4-bit `stq` field pointing into the Speculative Tile-Store Queue.

**Summary — entries vs. area (v2):**

| RS | Entries | Entry width (v2) | Storage | Comparators | Dominant sizing factor |
|----|---------|-----------------|---------|-------------|----------------------|
| Scalar | 32 | ~178 b (+8 b btag/ssb) | ~712 B | 384 (7b × 6 CDB) | Multi-cycle MUL/DIV latency + DIV blocking |
| LSU | 24 | ~178 b | ~534 B | 288 (7b × 6 CDB) | L1 miss latency (MLP) + 8 MSHRs + SSB |
| Vector | **24** | **~92 b** | **~276 B** | 288 (8b × 4 TCB × 3 srcs) | 16-cy epoch latency + multi-cycle PTOs + dispatch bursts |
| Cube | 4 | ~92 b | ~46 B | 16 (8b × 4 TCB) | Infrequent instructions, 1 in-flight |
| MTE | 16 | ~92 b | ~184 B | 64 (8b × 4 TCB) + 96 (7b × 6 CDB) | TILE.LD MLP + 7 write ports + STQ |
| **Total** | **100** | — | **~1752 B** | **~1136** | |

### 7.2 Reservation Station Entry Format

**Scalar / LSU RS entry (v2):**

```
 ┌─────────┬──────┬─────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┬─────┬─────┐
 │ valid   │ age  │ btag│ op   │ psrc1│ rdy1 │data1 │ psrc2│ rdy2 │data2 │ pdst  │ ckpt│ ssb │
 │ (1b)    │ (6b) │(3b) │(8b)  │(7b)  │(1b)  │(64b) │(7b)  │(1b)  │(64b) │ (7b)  │(3b) │(5b) │
 └─────────┴──────┴─────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴───────┴─────┴─────┘
   ~178 bits per scalar/LSU entry (v1: ~170; +8 b for branch_tag and SSB index)
```

**Tile-domain RS entry (v2 — Vector / Cube / MTE):**

```
 ┌──────┬──────┬─────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬─────┬─────┬──────┐
 │valid │ age  │ btag│ op   │ptsrc1 │trdy1 │ptsrc2 │trdy2 │ptsrc3 │trdy3 │ptdst1 │ptdst2│pscalar│ ckpt│ stq │meta_v│
 │(1b)  │(6b)  │(3b) │(8b)  │ (8b)  │(1b)  │ (8b)  │(1b)  │ (8b)  │(1b)  │ (8b)  │ (8b) │(8b)*  │(3b) │(4b) │(1b)  │
 │      │      │     │      │ +xp   │      │ +xp   │      │ +xp   │      │       │      │       │     │     │      │
 │      │      │     │      │ (1b)  │      │ (1b)  │      │ (1b)  │      │       │      │       │     │     │      │
 └──────┴──────┴─────┴──────┴───────┴──────┴───────┴──────┴───────┴──────┴───────┴──────┴───────┴─────┴─────┴──────┘
   ~92 bits per tile-domain entry (v1: ~80)
   (* pscalar field is double-wide: 7 b GPR tag + 1 b GPR-vs-IMM selector;
    sized to carry up to 2 scalar GPR tags for VEC-4K-v2 SX/SY operands
    encoded compactly in `pscalar` + immediate field of `op`)
```

New fields vs. v1:

| Field | Width | Purpose |
|-------|-------|---------|
| `btag` | 3 b | Branch tag of the youngest unresolved branch ahead of this µop. Used for one-cycle CAM-clear on mispredict. |
| `ptsrc3 + xp` | 8+1 b | Third source tile (the `C` mask in VEC, or 3rd cube operand). xp = is_xpose forwarded to TRegFile. |
| `xp_A`, `xp_B`, `xp_C` | 1 b each | Per-operand TRegFile-side `is_transpose` bits (forwarded to TRegFile read port at issue) |
| `ptdst2` | 8 b | Second destination tile (for dual-retire ops like `TMRGSORT`, `TROWARGMAX`) |
| `ssb` | 5 b | Speculative-Store-Buffer index (LSU stores only) |
| `stq` | 4 b | Speculative Tile-Store Queue index (MTE bulk stores only) |
| `meta_v` | 1 b | Metadata-valid flag (set when the producer's metadata write has propagated) |

### 7.3 Wakeup Logic

> **(v1 → v2: CDB 端语义未变,完整复制 v1 §7.3 描述。TCB 端因 VEC-4K-v2 第三源操作数被加宽,vector RS 比较器数量翻倍。)**

When an execution unit broadcasts a result on the **Common Data Bus (CDB)**, every reservation station entry compares its source tags against the CDB tag (v1 §7.3, 未变更):

```
  CDB broadcast: (tag=P40, data=0x1234)

  For each RS entry:
    if (psrc1 == P40 && !rdy1):  rdy1 ← 1;  data1 ← 0x1234
    if (psrc2 == P40 && !rdy2):  rdy2 ← 1;  data2 ← 0x1234

  Hardware: N entries × 2 sources × 7-bit comparators × C CDB ports
  Scalar RS: 32 × 2 × 6 = 384 comparators (CDB has 6 write-back ports)
```

An instruction becomes **ready to issue** when `rdy1 && rdy2` (both operands available).

**TCB wakeup (v2 widened):** 4 TCB ports broadcast 8 b tile tags. Tile-domain RS entries compare against ptsrc1/ptsrc2/ptsrc3 (3 sources × 4 ports = 12 comparators per entry; for the v2 24-entry vector RS this is 288 comparators, vs. v1's 16-entry × 2-source × 4 = 128).

### 7.4 Select Logic

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §7.4。)**

Each functional unit's select logic picks the **oldest ready** instruction from its reservation station every cycle:

```
  Select priority: lowest age value among entries with (valid && rdy1 && rdy2)

  Per cycle:
    Scalar RS → select up to 4 ALU + 1 MUL + 1 BRU (6 instructions)
    LSU RS    → select 1 load + 1 store
    Vector RS → select 1 vector op
    Cube RS   → select 1 cube op
    MTE RS    → select up to 2 tile ops
```

**Issue conflicts:** If multiple ready instructions target the same functional unit type and only one slot is available, the oldest wins. Younger instructions remain in the RS for the next cycle.

### 7.5 Dispatch Stall Conditions (v2 additions)

| Condition | Recovery |
|-----------|----------|
| Target RS is full | Wait for RS entry to be freed |
| Scalar / Tile free list empty | Wait for refcount-driven free |
| All checkpoint / branch-tag slots occupied | Wait for an in-flight branch to resolve |
| **SSB full** (LSU stores blocked) | Wait for SSB entry to drain (oldest non-speculative store commits) |
| **STQ full** (MTE bulk stores blocked) | Wait for STQ entry to drain |

---

## 8. Execution Units

### 8.1 Scalar Unit

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §8.1。)**

The scalar unit contains **6 functional units** sharing the Scalar RS.

#### 8.1.1 ALU (×4) — (v1 §8.1.1, 未变更)

Four identical single-cycle ALUs handle integer arithmetic, logic, shift, and compare operations.

| Parameter | Value |
|-----------|-------|
| Count | **4** symmetric ALUs |
| Operations | ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU, LUI, AUIPC |
| Latency | **1** cycle |
| Throughput | **4** ops / cycle |
| Input width | 64-bit |

#### 8.1.2 MUL/DIV Unit (×1) — (v1 §8.1.2, 未变更)

| Parameter | Value |
|-----------|-------|
| MUL latency | **4** cycles (pipelined, 1 MUL issued/cycle) |
| MUL operations | MUL, MULH, MULHU, MULHSU, MULW |
| DIV latency | **12–20** cycles (non-pipelined, blocks MUL during execution) |
| DIV operations | DIV, DIVU, REM, REMU, DIVW, DIVUW |

#### 8.1.3 Branch Unit (×1) — (v1 §8.1.3, 未变更)

| Parameter | Value |
|-----------|-------|
| Latency | **1** cycle (compare + resolve) |
| Operations | BEQ, BNE, BLT, BGE, BLTU, BGEU, JAL, JALR |
| On correct prediction | Deallocate checkpoint; no pipeline impact |
| On mispredict | Flash-restore RAT; flush pipeline stages F1–IS; redirect fetch |
| Mispredict penalty | **6** cycles (front-end refill) |

### 8.2 Load/Store Unit (LSU)

> **(v1 → v2: §8.2.1 架构与 §8.2.2 参数完整复制自 v1。v2 用 SSB §8.2.3 替换 v1 §8.2.3 简化提交。)**

The LSU handles all scalar memory operations with a **simplified** design enabled by the no-exception guarantee. The LSU pipeline is identical to v1 in terms of address calculation, TLB access, cache lookup, and L1-D MSHRs. The **store path** is the only structural change: stores no longer commit directly to L1-D; instead, they pass through a **Speculative Store Buffer** (SSB) that gates them by branch tag.

#### 8.2.A Architecture (v1 §8.2.1, 未变更)

```
  ┌──────────────────────────────────────────────────────────┐
  │  Load/Store Unit                                          │
  │                                                          │
  │  LSU RS (24 entries) ──┬──▶ Load Pipeline  (EX1–EX4)    │
  │                        └──▶ Store Pipeline (EX1–EX4)    │
  │                                                          │
  │  ┌─────────────────┐    ┌─────────────────┐             │
  │  │ Load Queue       │    │ Store Buffer →  │             │
  │  │ (16 entries)     │    │  SSB (24 ent.)  │  ← v2       │
  │  │ addr + tag       │    │ addr + data +   │             │
  │  │                  │    │ branch_tag      │             │
  │  └────────┬────────┘    └────────┬────────┘             │
  │           │  store-to-load       │                       │
  │           │◀─ forwarding ────────┘                       │
  │           ▼                      ▼                       │
  │      ┌──────────────────────────────┐                    │
  │      │        L1-D Cache (64 KB)    │                    │
  │      │    4-way, 64B line, 8 MSHRs  │                    │
  │      └──────────────────────────────┘                    │
  └──────────────────────────────────────────────────────────┘
```

#### 8.2.B Key Parameters (v1 §8.2.2, 未变更 except store buffer entries)

| Parameter | v1 | v2 |
|-----------|----|----|
| Load pipeline latency | **4** cycles (address calc + TLB + cache access + align) | **4** cycles (unchanged) |
| Store pipeline latency | **4** cycles (address calc + TLB + write to store buffer) | **4** cycles (unchanged) |
| Load queue entries | **16** | **16** (unchanged) |
| Store buffer entries | **16** | **24** (now SSB §11.4) |
| Store-to-load forwarding | Full forwarding when address and size match | Full forwarding (now from SSB §11.4.3) |
| L1-D MSHRs | **8** (non-blocking, 8 outstanding misses) | **8** (unchanged) |
| D-TLB | 64 entries, fully associative | unchanged |

#### 8.2.1 Speculative Store Buffer (SSB) — overview

```
  ┌────────────────────────────────────────────────────────────┐
  │  Speculative Store Buffer  (SSB, 24 entries)              │
  │                                                            │
  │  Each entry:                                               │
  │   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐       │
  │   │valid │ btag │ addr │ data │ size │ alloc│ drain│       │
  │   │ (1b) │ (3b) │ (40b)│(128b)│ (3b) │_age  │_rdy  │       │
  │   │      │      │      │      │      │ (6b) │ (1b) │       │
  │   └──────┴──────┴──────┴──────┴──────┴──────┴──────┘       │
  │                                                            │
  │   Total: 24 × ~182 b ≈ ~550 B                              │
  └────────────────────────────────────────────────────────────┘
```

| Field | Width | Purpose |
|-------|-------|---------|
| `valid` | 1 b | Slot is occupied |
| `btag` | 3 b | Branch tag inherited from the producing store µop |
| `addr` | 40 b | Physical address (post-TLB) |
| `data` | 128 b | Up to 16 B of store data (covers byte/half/word/dword) |
| `size` | 3 b | 1/2/4/8/16 B store width |
| `alloc_age` | 6 b | Sequence number for in-order drain |
| `drain_rdy` | 1 b | Set when `btag = 0xFF` (non-speculative); the entry can drain to L1-D |

**Capacity:** 24 entries — a 50% increase over v1's 16-entry "store buffer". The increase is driven by the speculation window: at maximum, all 8 branch tags can have their stores in flight, and each branch may generate several stores. With branch-prediction accuracy ~95% and a typical kernel mix of 20–30% memory ops, 24 entries provide ~10 cycles of buffering at peak issue (2 stores/cycle into a smaller pool would risk dispatch stall).

#### 8.2.2 SSB drain policy

```
  Tag-clear from speculation tracker (when branch resolves correctly):
    For each SSB entry e:
      if e.btag == cleared_tag:
         e.btag ← (next-older unresolved branch tag, or 0xFF if none)
         if e.btag == 0xFF: e.drain_rdy ← 1

  Drain to L1-D (1 store/cycle, oldest-first among drain_rdy entries):
    Pick oldest e with valid && drain_rdy
    Issue write to L1-D pipeline (with 4-cy occupancy, like v1 store)
    On completion: e.valid ← 0; SSB entry returned to free pool

  Mispredict (tag invalidation):
    For each SSB entry e:
      if e.btag is younger than (or equal to) mispredicted_tag:
         e.valid ← 0   (entry invalidated; never reaches L1-D)
```

The SSB head pointer is the **next allocation slot** (FIFO discipline for in-flight ordering); the head is snapshotted into each branch checkpoint at D2.

#### 8.2.3 Store-to-load forwarding from SSB

Loads still forward from the SSB on address match (same semantic as v1's store-to-load forwarding). Only loads with the **same or younger** branch_tag are eligible to forward — a load on a different speculation path must NOT forward from a store on its own path; instead, the load goes to L1-D.

```
  Load (addr, btag_load) forwards from SSB entry e iff:
    e.valid && e.addr == addr && e.size matches &&
    e.btag is "ancestor" of btag_load
            ──────────────────────
            i.e. e.btag is in the chain of unresolved branches
            that branch_tag_load also depends on.
```

Implementation: the speculation tracker (§11.3) maintains a 8 × 8 ancestry bitmap; load-forwarding eligibility is one bitmap-lookup + AND.

#### 8.2.4 SSB area

| Block | Area |
|-------|------|
| 24 × 182 b flip-flop array | ~24 × 1.8 K gate ≈ ~45 K gate |
| Address CAM (24 × 40 b for forwarding) | ~30 K gate |
| Branch-tag ancestry bitmap (8 × 8 b) | ~1 K gate |
| Drain FSM | ~2 K gate |
| **Total** | **~80 K gate** (~0.02 mm² @ 5 nm) |

### 8.3 Vector Unit — VEC-4K-v2

The vector unit is a full re-architecting of v1's vector unit, specified in detail in [`vector4k_v2.md`](vector4k_v2.md). This section summarizes the integration into the Davinci-v2 core; refer to the standalone document for full datapath, microcode, and worked microcode examples.

#### 8.3.1 High-level structure

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  VEC-4K-v2 Unit                                                       │
  │                                                                      │
  │  TRegFile-4K read ports R0, R4 (with is_transpose)                   │
  │       │  512 B/cy each                                               │
  │       ▼                                                              │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                            │
  │  │ SA       │  │ SB       │  │ SC       │  ← staging registers       │
  │  │ 4 KB     │  │ 4 KB     │  │ 4 KB     │     (24 × 1R1W SRAM        │
  │  │ SRAM     │  │ SRAM     │  │ SRAM     │      macros total, §9.1    │
  │  │ +meta    │  │ +meta    │  │ +meta    │      of vector4k_v2.md)    │
  │  └─────┬────┘  └─────┬────┘  └────┬─────┘                            │
  │        │ 512 B/cy    │ 512 B/cy   │ 1-bit-per-lane mask              │
  │        ▼             ▼            ▼                                  │
  │  ┌───────────────────────────────────────────────────────────┐       │
  │  │  Stage (A): align / unpack / permute (per operand slot)    │       │
  │  └─────────────────────────────┬───────────────────────────────┘       │
  │                                ▼                                      │
  │  ┌───────────────────────────────────────────────────────────┐       │
  │  │  Stage (B): 128 compute groups (FP32-equivalent FMA core   │       │
  │  │             with intra-group format widener), Acc feedback,│       │
  │  │             per-lane mask gate, 256-lane shuffle+CAS       │       │
  │  │             primitive (for TMRGSORT)                        │       │
  │  └───────────────────────────────────────────────────────────┘       │
  │                                                                       │
  │  ┌───────────────────────────────────────────────────────────┐       │
  │  │  Acc (256 × 32 b × 2 ping-pong, parity-banked LO/HI)       │       │
  │  └───────────────────────────────────────────────────────────┘       │
  │                                                                       │
  │  ┌──────────────┐    ┌──────────────┐                                 │
  │  │ Pack → D0    │    │ Pack → D1    │                                 │
  │  │ → W0 (512B/cy)│    │ → W4 (512B/cy)│                               │
  │  └──────────────┘    └──────────────┘                                 │
  │                                                                       │
  │  Microcode ROM: ~64 b × 64 beats × 256 programs ≈ 128 Kb              │
  └──────────────────────────────────────────────────────────────────────┘
```

#### 8.3.2 Key parameters

| Parameter | Value | Note |
|-----------|-------|------|
| Compute datapath width | 512 B / 128-lane FP32-equivalent | unchanged from v1 |
| TRegFile read ports used | R0 (Port A), R4 (Port B) | per epoch; 8 cy/epoch |
| TRegFile write ports used | W0 (D0), W4 (D1) | per epoch |
| Staging: SA, SB, SC payload | 4 KB each; 1R1W SRAM (recommended) | 24 × `512 b × 16 × 1R1W` macros |
| Compute groups | 128 (format-independent) | Per-format SIMD width: FP32×1, FP16×2, FP8×4, FP4×4(×2 sub-beats) |
| Accumulator | 256 × 32 b × 2 ping-pong, LO/HI parity-banked | `N_run = 512` |
| Microcode ROM | ~128 Kb (64 b × ~64 beats × ~256 programs) | regenerable per ISA version |

#### 8.3.3 Tile metadata flow

Each VEC-4K-v2 instruction at D2 rename consults the **Tile Metadata RAT** (§6.1) for each source tile and forwards `(shape.x, shape.y, format)` to the VEC RS entry. At issue, the metadata is propagated into the `SOP` staging register ([`vector4k_v2.md`](vector4k_v2.md) §4.4) where it stays stable for the entire compute phase. The destination tile's metadata is **derived** from the operation's semantics and `retire_format_*` fields (e.g. `VCVT` produces metadata with `format = retire_format_0`, same shape as source).

For backward-compatibility with v1 binaries, a tile that has never been written by a v2 instruction (and therefore has no explicit metadata) defaults to `(shape.x, shape.y, format) = (16, 64, FP32)` — the canonical v1 4 KB tile interpretation.

#### 8.3.4 Microcode beat machine

Compute is driven beat-by-beat from a microcode ROM (`SOP.ucode_base`, `SOP.ucode_len`). Each beat word (~64 b) names per ALU operand slot:

- `src` ∈ {SA, SB, ACC_READ_LO, ACC_READ_HI, SX_broadcast, SY_broadcast, IMM_ZERO}
- `s` (strip index 0..7)
- `tilelet_xpose` (per-beat, primary transpose mechanism)
- `mask_src` ∈ {SC_mask, IMM_ALL_ONES, IMM_FROM_SOP}
- `alu_op` ∈ {ADD, SUB, MUL, FMA, FNMA, MAX, MIN, CMP, AND, XOR, PASS_A, PASS_B, SELECT, RECIP, RSQRT, SHUFFLE_CAS_UP, SHUFFLE_CAS_DOWN, …}
- `acc_op` ∈ {NONE, INIT, ACCUM, MERGE_STAGE, READOUT}
- `wr_en_D0`, `wr_en_D1`

Microcode is keyed by `(opcode, format, W-regime, R-regime)`; the ROM is regenerable in software per ISA version (no RTL change).

#### 8.3.5 Operand-fetch prologue

Operand fetch is **variable-length** ([`vector4k_v2.md`](vector4k_v2.md) §6):

| `N_val` | `is_xpose` mix | `has_mask` | `T_fetch` (best/worst) |
|--------:|----------------|-----------:|------------------------:|
| 1 | any | 0 or 1 | **8 / 15 cy** |
| 2 | uniform | 0 or 1 | **8 / 15 cy** |
| 2 | mixed (R2 penalty) | 0 or 1 | **16 / 23 cy** |

The mask fetch (1–2 strips of `SC`) piggybacks on an idle port cycle within a value-tile epoch and never extends the prologue in the common case.

#### 8.3.6 Per-operand `is_transpose` and per-beat `tilelet_xpose`

VEC-4K-v2 forwards each operand's `is_xpose` bit (latched at D2 rename, present in the RS entry §7.2) to the assigned TRegFile read port at issue time. The TRegFile delivers either row-mode or col-mode strips; the staged content reflects the requested mode.

Per-beat `tilelet_xpose` (microcode bit per operand slot per beat) re-transposes the staged tile inside the staging register's diagonal-skew read datapath at no scheduling cost. Most reduction kernels use `tilelet_xpose` exclusively (TRegFile-side bit defaults to 0); the TRegFile-side bit is reserved for cases where the same transposed view is reused many times.

#### 8.3.7 Latency table (selected ops)

| Op | Format | Shape | Latency |
|----|--------|-------|---------|
| VADD / VMUL / VFMA_ACC (binary or Acc-feedback ternary) | any | 1024..8192 elements | 16 cy (8 fetch + 8 retire); throughput 1/8 cy |
| **VFMA / VFNMA** (native 3-source, `c_role=VALUE`, §2.2.6a) | any | 1024..8192 elements | **16 cy (8 fetch on R0/R4/R1 + 8 retire); throughput 1/8 cy** — same as binary thanks to 3-port parallel fetch |
| **VLERP** (native 3-source, dual retire D0/D1, §2.2.6a) | any | 1024..8192 elements | **24 cy (8 fetch + 16 retire); throughput 1/16 cy** |
| VADD masked | any | any | same as unmasked (mask piggybacks) |
| VFMA_ACC with `is_xpose_A ≠ is_xpose_B` | any | any | **24 cy** (16 fetch + 8 retire); throughput 1/16 cy |
| **VFMA with one mismatched `is_xpose_*`** | any | any | **24 cy** (16 fetch + 8 retire); throughput 1/16 cy |
| **VFMA with all three `is_xpose_*` distinct** (degenerate) | any | any | **32 cy** (24 fetch + 8 retire); throughput 1/24 cy |
| VROWSUM (wide, R=8 C=128 FP32) | FP32 | 8×128 | 16 cy fetch + 13 compute + 8 retire ≈ **37 cy** (recommended baseline, no cross-group tree) |
| VROWSUM (alt config with cross-group tree) | FP32 | 8×128 | 16 + 9 + 8 = **33 cy** |
| VCOLSUM (wide) | FP32 | 8×128 | 8 + 9 + 8 = **25 cy** |
| TMRGSORT N=256 FP32 | FP32 | 256 elements | 8 + 36 + 8 = **52 cy** |
| TMRGSORT N=1024 FP32 (1 tile) | FP32 | 1024 elements | 8 + 220 + 8 = **236 cy** |
| TINV 32×32 FP32 (1 tile) | FP32 | 32×32 | 8 + ~2 200 + 8 = **~2.2 K cy** |
| TINV 128×128 FP32 (16 tiles) | FP32 | 128×128 | ~120 cy fetch + ~33 K + 8 retire = **~33 K cy** |
| TROWRANGE_MUL 8 strips | FP32 | 8×128 | 8 + 10 + 8 = **26 cy** |

#### 8.3.8 TRegFile-4K port usage

A single VEC-4K-v2 instruction occupies:

- 1–2 TRegFile read ports for `T_fetch` cycles (R0 + optional R4)
- 1–2 TRegFile write ports for the 8-cycle retire epoch (W0 + optional W4 if `retire_mask = 2'b11`)

**Epoch-pipelined throughput** is preserved: the retire epoch of instruction N overlaps with the fetch epoch of instruction N+1 on independent ports. The per-port allocation table (§9.2.4) is unchanged from v1 in topology — VEC-4K-v2 happens to use the same R0/R4 + W0/W4 binding as v1.

#### 8.3.9 Speculation handling

VEC-4K-v2 instructions are speculation-safe **inherently**: their only "external" side effect is a tile write to TRegFile-4K, which is renamed and reference-counted (§10.5). On branch mispredict:

- VEC RS entries with `btag` in the misprediction's flush set are invalidated.
- In-flight compute beats are flushed within 1 cycle (the staging-register state is overwritten by the next instruction's operand fetch).
- Destination physical tiles (allocated at D2) are returned to the tile free list via free-list-head-pointer restore from the checkpoint.

**No special speculation hardware is needed inside the vector unit.** The flush converges to a quiescent state in `T_fetch + max_beat_count` cycles in the worst case (a long-running `TINV` or `TMRGSORT` taking the full hit), but this is bounded by the number of in-flight vector ops (≤ 24 RS entries) and does not stall the front-end's recovery.

### 8.4 Cube Unit (outerCube MXU)

> **(v1 → v2: §8.4.1 / §8.4.2 完整复制自 v1。v2 增量见本节末:cube 现可消费 TRegFile-4K 的 col-mode 读出。)**

The cube unit is the outerCube Matrix Unit, a large-scale outer-product accumulation engine. Full specification is in [`outerCube.md`](outerCube.md).

#### 8.4.1 Summary (v1 §8.4.1, 未变更)

| Parameter | Value |
|-----------|-------|
| Base MAC units | **4096** (8 banks × 8 rows × 64 columns) |
| Modes | **Mode A** (K-parallel, 8-bank reduction) / **Mode B** (M-parallel, independent) |
| Formats | FP16, BF16, FP8 (E4M3/E5M2), MXFP4, HiFP4 |
| MAC scaling | FP16: 4096 / FP8: 8192 / MXFP4: 32768 MACs/cycle |
| Accumulator | 32-bit FP32, ping-pong (2 × 16 KB = 32 KB) |
| Pipeline | 19 stages: 8 (OF) + 1 (MUL) + 1 (RED) + 1 (ACC) + 8 (AD) |
| Staging SRAM | A double-buffer (8 KB) + B double-buffer (32 KB) = 40 KB baseline |
| Peak FP16 @ 1.5 GHz | **12.3 TFLOPS** |
| Peak FP8 @ 1.5 GHz | **24.6 TOPS** |
| Peak MXFP4 @ 1.5 GHz | **98.3 TOPS** |

#### 8.4.2 Cube Instruction Dispatch (v1 §8.4.2, 未变更)

Cube instructions (CUBE.OPA, CUBE.DRAIN, etc.) are dispatched to the **Cube RS** (4 entries) after Tile RAT rename. Each CUBE.OPA is a long-running instruction that occupies the MXU for many cycles (N + 18, where N = Nb × S OPA steps). While the MXU is busy, the Cube RS holds subsequent cube instructions until the current one completes.

A CUBE.OPA may reference a range of architectural tile registers (e.g., T[Tb]..T[Tb+Na−1]). At dispatch, the Tile RAT translates each architectural tile index to a physical tile index. For multi-tile operands, the cube RS stores a base physical tile index plus a **tile address table** (up to 16 entries) holding the physical indices of all tiles in the range; the cube pipeline controller uses these physical indices to program TRegFile-4K port addresses.

The cube unit reads tile data from TRegFile-4K ports R0 (A operand) and R1–R4 (B operand), and drains results via W0 (C output). Port interactions are managed by the cube pipeline controller, which issues epoch-aligned physical tile addresses to TRegFile-4K's pending registers (see [`tregfile4k.md`](tregfile4k.md) §3).

#### 8.4.3 v2 增量(对硬件不可见)

The cube unit benefits indirectly from the TRegFile-4K `is_transpose` enhancement: software can now feed the cube either row-major or col-major B-operand tiles by setting `is_xpose` on the cube's B-operand tile-RAT entries (the cube pipeline controller propagates the bit to TRegFile read ports R1–R4), eliminating the need for `TILE.TRANSPOSE` predecessors in many GEMM kernels. The cube ALU, accumulator, and pipeline are otherwise unchanged.

### 8.5 MTE Unit

> **(v1 → v2: §8.5.A / §8.5.B / §8.5.C / §8.5.D 完整复制自 v1 §8.5.1 / §8.5.2 / §8.5.3 / §8.5.4 / §8.5.5。v2 增量集中在 §8.5.1 (TRANSPOSE 缩减) 与 §8.5.2 (STQ)。)**

The MTE unit is the **bridge between three domains**: memory ↔ TRegFile-4K (bulk tile transfers) and scalar GPR ↔ TRegFile-4K (single-element access via TILE.GET/TILE.PUT). All MTE instructions go through full **dual-RAT rename** at D2: scalar operands are renamed via the Scalar RAT, and tile operands are renamed via the Tile RAT. Instructions that produce a new tile (TILE.LD, TILE.ZERO, TILE.COPY, TILE.GATHER, TILE.PUT) allocate a fresh physical tile from the tile free list. TILE.GET produces a scalar GPR result and broadcasts on the CDB.

#### 8.5.A Architecture (v1 §8.5.1, 未变更)

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Memory Tile Engine (MTE)                                        │
  │                                                                  │
  │  MTE RS (16 entries) ──┬──▶ Load Tile Pipeline                  │
  │                        ├──▶ Store Tile Pipeline ──▶ STQ (v2)    │
  │                        ├──▶ Gather Pipeline                     │
  │                        ├──▶ Scatter Pipeline ──▶ STQ (v2)       │
  │                        ├──▶ TILE.GET Pipeline (tile→GPR)        │
  │                        └──▶ TILE.PUT Pipeline (GPR→tile, RMW)   │
  │                                                                  │
  │  ┌──────────────────────────────┐                                │
  │  │ Outstanding Request Buffer   │  Tracks up to 32 in-flight    │
  │  │ (32 entries)                 │  tile transfers for MLP        │
  │  └──────────────┬───────────────┘                                │
  │                 │                                                │
  │  ┌──────────────▼───────────────┐  ┌─────────────────────────┐  │
  │  │ Address Generation Unit      │  │ Data Assembly / Scatter  │  │
  │  │ (contiguous, strided, index) │  │ (pack / unpack for G/S)  │  │
  │  └──────────────┬───────────────┘  └──────────┬──────────────┘  │
  │                 │                              │                 │
  │                 ▼                              ▼                 │
  │  ┌──────────────────────────────────────────────────┐           │
  │  │  L2 / Memory Interface (high-bandwidth path)     │           │
  │  │  64 B/cy (1 cache line/cy) sustained              │           │
  │  └──────────────────────────────────────────────────┘           │
  │                 │                              │                 │
  │                 ▼                              ▼                 │
  │  ┌──────────────────────────────────────────────────┐           │
  │  │  TRegFile-4K Write Ports (W1–W7 for TILE.LD)    │           │
  │  │  TRegFile-4K Read Ports (R5–R7 for TILE.ST)     │           │
  │  └──────────────────────────────────────────────────┘           │
  │                                                                  │
  │  ┌──────────────────────────────────────────────────┐           │
  │  │  Scalar GPR ↔ Tile Element Path                  │           │
  │  │  TILE.GET: TRegFile read port → extract → CDB    │           │
  │  │  TILE.PUT: CDB snoop → tile copy + insert → write│           │
  │  └──────────────────────────────────────────────────┘           │
  └──────────────────────────────────────────────────────────────────┘
```

#### 8.5.B Key Parameters (v1 §8.5.2, 未变更)

| Parameter | Value |
|-----------|-------|
| TILE.LD TRegFile write | **8 cycles** per write port (512 B/cy × 8 cy = 4 KB) |
| TILE.LD total latency (L2 hit) | **72 cycles** (64 cy memory fetch + 8 cy TRegFile write epoch) |
| TILE.ST TRegFile read | **8 cycles** per read port (512 B/cy × 8 cy = 4 KB) |
| TILE.ST total latency (L2) | **72 cycles** (8 cy TRegFile read epoch + 64 cy memory write) |
| Available write ports | W1–W7 (**7** ports, minus ports used by cube drain) |
| Available read ports | R5–R7 (**3** ports, minus ports used by cube operands) |
| Max concurrent TILE.LD | up to **7** (1 per write port), limited by memory BW |
| Max concurrent TILE.ST | up to **3** (1 per read port) |
| Outstanding request buffer | **32** entries (supports deep memory-level parallelism) |
| Gather/scatter | Uses index tile (Tidx) for non-contiguous access patterns |
| L2 → MTE bandwidth | **64 B/cy** (1 cache line/cy) → 1 tile in **64 cycles** from L2 |
| TILE.COPY / TILE.TRANSPOSE latency | **16 cycles** (8 cy TRegFile read epoch + 8 cy write epoch) |
| TILE.ZERO latency | **8 cycles** (1 write epoch, no read needed) |
| **TILE.GET latency** | **9 cycles** (8 cy TRegFile read epoch + 1 cy element extract → CDB) |
| **TILE.PUT latency** | **16 cycles** (8 cy read epoch + 8 cy write epoch); **8 cy** with copy elision |
| TILE.GET throughput | **1 per 8 cycles** (read port occupied for full epoch even for single element) |
| TILE.PUT throughput | **1 per 16 cycles** (read + write port, 2 epochs); **1 per 8 cy** with elision |

#### 8.5.C MTE Rename → Issue → Execute Flow (Bulk Transfer) — (v1 §8.5.3, 未变更)

```
  D2 (Rename):
    TILE.LD T10, [X5]
      Scalar RAT: X5 → P40 (physical scalar for base address)
      Tile RAT:   T10 → PT200 (allocate new physical tile from tile free list)
                  old mapping PT10 marked orphan
      Tile RAT ready[PT200] ← 0

  DS (Dispatch):
    MTE RS entry: {op=TILE.LD, pscalar=P40, srdy=<from Scalar RAT>, ptdst=PT200, ckpt=...}

  IS (Issue):
    Wait for pscalar P40 ready (CDB wakeup from scalar ALU)
    → read base address from scalar physical RF

  EX (Execute — memory fetch + 1 TRegFile write epoch):
    Memory phase (≈64 cycles from L2):
        MTE Address Gen: compute contiguous address range from base address
        MTE Data Path:   request 64 cache lines from L2 (64 B/cy)
        MTE Buffer:      accumulate 4 KB in outstanding request buffer
    TRegFile write epoch (8 cycles):
        Reserve write port, program reg_idx = PT200
        Write 512 B/cy × 8 cy = 4 KB to physical tile slot PT200
    Total TILE.LD latency (L2 hit): 64 + 8 = **72 cycles**

  Complete:
    Tile RAT ready[PT200] ← 1
    TCB broadcast: PT200
    → wake dependent instructions in Vector RS, Cube RS, MTE RS
    Decrement tile refcount for any source tiles
```

MTE bulk operations incur both **memory latency** and **TRegFile epoch latency**. For TILE.LD, the MTE first fetches 4 KB from memory (64 cache lines at 64 B/cy = 64 cycles from L2), buffers the data, then writes to TRegFile-4K in one 8-cycle write epoch using the **physical tile index** (from Tile RAT) as the `reg_idx` address — total latency: **memory + 8 cycles**. For TILE.ST, the MTE first reads the tile from TRegFile in one 8-cycle read epoch, then writes the data to memory — total: **8 cycles + memory**. The MTE controller issues physical `reg_idx` addresses to port pending registers and sequences data transfer across each 8-cycle epoch.

#### 8.5.D TILE.GET / TILE.PUT Execution Flow (Element Access) — (v1 §8.5.4, 未变更)

**TILE.GET Rd, Ts, Ridx** — scalar ← tile element:

```
  D2 (Rename):
    Scalar RAT: Ridx → P50 (lookup index);  Rd → P60 (allocate new scalar dest)
    Tile RAT:   Ts → PT180 (lookup source tile)

  DS (Dispatch):
    MTE RS entry: {op=TILE.GET, pscalar=P50(Ridx), srdy, pdst=P60(Rd), ptsrc1=PT180(Ts), trdy}

  IS (Issue):
    Wait for P50 ready (CDB wakeup) AND PT180 ready (TCB wakeup)
    → read index value from scalar RF; compute row_group = row / 8, row_off, col

  EX (Execute, 9 cycles):
    Cycles 1–8: TRegFile read epoch — reserve read port for physical tile PT180
                port reads 512 B/cy × 8 cy (full tile streamed out);
                capture the 512-B chunk at cycle (row_group+1) containing target row
    Cycle 9:    extract element from captured 512-bit row based on col and
                funct3 (element type), zero-extend to 64 bits

  Complete:
    CDB broadcast: (tag=P60, data=element_value)
    → wakeup dependent scalar RS entries; write to scalar physical RF
    Decrement tile refcount for PT180
```

**TILE.PUT Td, Rs, Ridx** — tile element ← scalar (read-modify-write):

```
  D2 (Rename):
    Scalar RAT: Rs → P70 (lookup data), Ridx → P71 (lookup index)
    Tile RAT:   Td old mapping → PT180 (source, for tile copy)
                Td new mapping → PT210 (allocate from tile free list)
                PT180 marked orphan; ready[PT210] ← 0

  DS (Dispatch):
    MTE RS entry: {op=TILE.PUT, pscalar=P70(Rs), pscalar2=P71(Ridx),
                   ptsrc1=PT180(Td_old), ptdst=PT210(Td_new)}

  IS (Issue):
    Wait for P70, P71 ready (CDB) AND PT180 ready (TCB)

  EX (Execute, 16 cycles — 2 full TRegFile epochs):
    Read epoch (cycles 1–8):
        Reserve read port for physical tile PT180
        Read 512 B/cy × 8 cy = 4 KB (full source tile)
        Buffer tile data in MTE internal SRAM; overwrite target element
        at (row, col) derived from Ridx with scalar value from Rs
    Write epoch (cycles 9–16):
        Reserve write port for physical tile PT210
        Write modified tile 512 B/cy × 8 cy = 4 KB to PT210

    Copy elision optimisation (8 cycles):
        When PT180 refcount=0 and is orphaned at rename, the copy is
        skipped. PT210 reuses PT180's storage. Only the target element
        is overwritten in-place during a single write epoch (8 cy).

  Complete:
    Tile RAT ready[PT210] ← 1
    TCB broadcast: PT210
    → wake dependent tile-domain RS entries
    Decrement tile refcount for PT180; if orphan and refcount=0 → free PT180
```

TILE.GET occupies a TRegFile read port for a full 8-cycle epoch (even though only one 512-B chunk is needed), plus 1 cycle for element extraction — **9 cycles** total. TILE.PUT requires two full epochs (8 cy read + 8 cy write = **16 cycles**) because it is a read-modify-write on the tile. With copy elision (PT_old orphaned, refcount=0), the read epoch is skipped and only the write epoch is needed — reducing latency to **8 cycles**.

#### 8.5.E TILE.MOVE (Move Elimination) — (v1 §8.5.5, 未变更)

**TILE.MOVE Td, Ts** — Handled entirely at the D2 rename stage with **zero-cycle latency**:

```
  D2 (Rename):
    TILE.MOVE T5, T10
      Tile RAT[T10] → PT180 (source physical tile)
      Tile RAT[T5]  → PT50  (old destination mapping, marked orphan)
      Tile RAT[T5]  ← PT180 (Td now aliases same physical tile as Ts)
      refcount(PT180) += 1   (extra architectural name)
      ready[T5] = ready[PT180]  (inherit readiness)
      → No RS entry allocated. No execute. No TCB broadcast.
      → Instruction completes immediately at D2.

  If PT50 is orphan and refcount==0 → free PT50 to tile free list
```

TILE.MOVE does not consume any execute-stage resources, TRegFile-4K ports, or memory bandwidth. It is the preferred way to "rename" tiles between software pipeline stages (e.g., double-buffering schemes where the next iteration's input tiles become the current iteration's operand tiles). Because Td and Ts share the same physical tile after TILE.MOVE, the next write to either architectural register will naturally allocate a new physical tile at rename time.

---

**v2 增量(下面 §8.5.1 / §8.5.2 / §8.5.3 / §8.5.4):**

#### 8.5.1 `TILE.TRANSPOSE` — reduced footprint

Because TRegFile-4K can deliver col-mode reads directly (§9.2), most "transpose then consume" patterns are subsumed by the consumer's `is_xpose` bit. The dedicated 4 KB MTE transpose buffer of v1 shrinks to a small **512 B element-level fixup buffer** used only for the non-aligned `W ∈ {128, 256, 1024, 2048, 4096}` regimes that [`tregfile4k.md`](tregfile4k.md) §7.5 leaves to downstream consumers. (For these regimes, the chunk-grid transpose at 64 B granularity is not element-level valid, and `TILE.TRANSPOSE` materializes an element-correct transpose in a new physical tile.)

| MTE TILE.TRANSPOSE behaviour | v1 | v2 |
|------------------------------|-----|-----|
| 4 KB transpose buffer | required | replaced by 512 B fixup SRAM |
| Latency | 16 cy | 16 cy (unchanged) |
| Use case | universal | rare — only when materializing a transposed tile for reuse across many instructions that don't carry `is_xpose` |

#### 8.5.2 Speculative Tile-Store Queue (STQ)

`TILE.ST` and `TILE.SCATTER` allocate an STQ entry at dispatch (alongside the regular MTE RS entry):

```
  STQ entry (8 entries total):
    valid (1b) | btag (3b) | base_addr (40b) | tile_phys_idx (8b) |
    stride (40b) | scatter_idx_phys (8b) | size_log2 (3b) | drain_rdy (1b)
    + meta_v (1b)
```

Total STQ size: 8 × ~110 b ≈ ~110 B. Drain logic mirrors the SSB:

- **Tag clear** (branch resolves correctly): `btag` is updated; if it reaches `0xFF`, `drain_rdy` is set, and the STQ controller can issue the actual memory write.
- **Mispredict**: STQ entries with `btag` younger or equal to the mispredicted tag are invalidated. The corresponding tile data — still resident in TRegFile-4K — is freed via the normal Tile RAT refcount path; no memory write was issued.
- **Drain**: oldest `drain_rdy` entry begins streaming the tile from TRegFile through the MTE memory pipeline. Drain is overlapped with subsequent MTE operations.

**Why only 8 entries?** Bulk tile stores are infrequent compared to scalar stores: a typical kernel issues 1 `TILE.ST` per ~10–50 scalar stores. The 8 entries provide ~64 cycles of buffering at peak issue (1 TILE.ST/4 cy), enough to absorb a burst at the end of a kernel without causing dispatch stall.

**Why STQ is separate from SSB:** the TILE.ST data payload (4 KB) cannot reasonably be captured in the SSB's flip-flop register — it must remain resident in TRegFile-4K. The STQ stores only the *intent* (address + tile-pointer + branch_tag) and triggers the memory write on commit.

#### 8.5.3 STQ area

| Block | Area |
|-------|------|
| 8 × 110 b flip-flop array | ~10 K gate |
| Branch-tag ancestry check (shared with SSB) | ~0 (reuses SSB's bitmap) |
| Drain controller FSM | ~2 K gate |
| **Total** | **~12 K gate** (~0.003 mm² @ 5 nm) |

#### 8.5.4 MTE RS entry per instruction (v2 update)

The `TILE.ST` and `TILE.SCATTER` entries gain a 4 b STQ index. Other MTE instructions are unaffected.

| Instruction | STQ allocation | Notes |
|-------------|----------------|-------|
| TILE.LD, TILE.GATHER, TILE.ZERO | — | tile-write only, refcount-managed; no memory side effect, fully recoverable via Tile RAT |
| TILE.ST, TILE.SCATTER | **STQ slot** | held in STQ until non-speculative |
| TILE.COPY, TILE.MOVE, TILE.TRANSPOSE | — | tile-internal, fully recoverable |
| TILE.GET | — | scalar GPR result; recoverable via Scalar RAT + ref-count |
| TILE.PUT | — | tile-write (RMW); recoverable via Tile RAT |

---

## 9. Register Files

### 9.1 Scalar GPR Physical Register File

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §9.1。)**

| Parameter | Value |
|-----------|-------|
| Physical registers | **128** (P0–P127), 64-bit each |
| Total storage | 128 × 8 B = **1 KB** |
| Read ports | **12** (8 from rename lookup + 4 from issue/execute) |
| Write ports | **6** (4 ALU + 1 MUL/LSU + 1 TILE.GET), matched to CDB ports |
| Implementation | Flip-flop array (small enough for full-speed multi-port) |
| Bypass network | 6-source → 12-sink forwarding MUXes |

**Bypass network:** When a result is broadcast on the CDB in the same cycle that an issuing instruction reads the physical RF, the bypass network forwards the CDB data directly to the execution unit input, avoiding a 1-cycle read-after-write penalty.

**Register lifecycle:**

```
  Allocate:  free list dequeue → assigned as destination at D2
  Write:     execution unit writes result at WB stage
  Read:      issuing instructions read at IS stage (or snoop from CDB)
  Orphan:    a later instruction remaps the same architectural register
  Free:      orphan AND reference count = 0 → return to free list
```

### 9.2 TRegFile-4K (with per-port `is_transpose`)

The TRegFile-4K is the physical tile register file for vector, cube, and MTE. Full specification: [`tregfile4k.md`](tregfile4k.md). v2 highlights below.

#### 9.2.1 Tile metadata storage

A new **256 × 32 b SRAM** sits alongside the 1 MB tile data array, holding `(shape.x, shape.y, format)` per physical tile. Read ports: 4 (decode) + 1 (TCB completion). Write ports: 2 (retire + `TSETMETA`). Storage: 1024 B = ~10 K gate.

The metadata is **physically distinct** from the 4 KB tile payload but is read together at the **first strip** of an operand fetch (§4.4 of [`vector4k_v2.md`](vector4k_v2.md)) so that the consumer's microcode program can be selected based on `(format, R, C)` before the second strip arrives.

#### 9.2.2 Per-port `is_transpose` flag

Each of the 8 read ports (R0–R7) accepts a 1-bit `is_transpose` flag double-registered alongside the 8-bit `reg_idx`. The flag is latched at the epoch boundary and held constant for the entire 8-cycle epoch.

| `is_transpose` | Strip delivery | Bank pattern per cycle |
|----------------|----------------|------------------------|
| 0 (row-mode) | strip `s` = chunk-grid row `s` (contiguous bytes `s·512 .. s·512+511`) | 8 banks of one group |
| 1 (col-mode) | strip `s` = chunk-grid column `s` (8 × 64 B chunks across all 8 groups along the wrapped diagonal) | 1 bank per group |

Both modes deliver **512 B/cy** through the 8-cycle epoch — same throughput, no extra latency. The diagonal skew layout `bank_id = 8·g + ((l + g) mod 8)` is what makes col-mode bank-conflict-free.

**Scheduling rule R2** ([`tregfile4k.md`](tregfile4k.md) §6): the 8 active read ports of any 8-cycle epoch must share the same `is_transpose` value. Mixed-mode reads in the same epoch collide on the 1R-port SRAM banks. This rule is enforced at the dispatch / port-allocation stage; if a vector instruction's operand `is_xpose_A ≠ is_xpose_B`, the operand-fetch prologue automatically splits into two epochs (16 cy instead of 8 cy, §8.3.6).

#### 9.2.3 Hardware delta vs. v1 TRegFile-4K

| Component | v1 | v2 |
|-----------|----|----|
| 64 SRAM banks (256×512b each) | yes | yes (unchanged) |
| Diagonal skew bank map | (introduced together) | yes |
| Per-port pending+active address registers | 1 reg_idx × 2 | (1 reg_idx + 1 is_transpose) × 2 — adds **1 b × 2 × 8 ports = 16 FF** |
| Read-port datapath: bank-select mux | 1 option (row-mode) | **2 options (row OR col)** — small 2-to-1 mux per port + col-mode address generator (`bank_i = 8·i + (p+cy+i) mod 8`) |
| Read-port output rotator | 8-way 64 B (always active in row-mode) | **8-way 64 B, active only in row-mode**; bypassed in col-mode |
| Metadata SRAM (256 × 32 b, 4R/2W) | — | **+10 K gate** |
| Write-side | unchanged | unchanged |
| Latency / throughput per port | 1 reg_idx / 8 cy | 1 reg_idx + 1 is_transpose / 8 cy (same epoch) |

**Total v2 delta**: ~12 K gate (mostly metadata SRAM + col-mode address generator). The transpose capability adds **zero SRAM duplication** and **zero latency** to the basic read path.

#### 9.2.4 Port allocation

Port assignment across vector, cube, and MTE units is **identical to v1 §9.2** (table reproduced below for reference). The introduction of `is_transpose` does not change which port serves which client; it only changes the data delivery order on each read port.

| Port | Cube active — MXFP4/HiFP4 | Cube active — FP16/BF16/FP8 | Cube idle |
|------|----------------------------|------------------------------|-----------|
| R0 | Cube A (1 tile/epoch) | Cube A (1 tile/epoch) | VEC-4K-v2 / MTE — free |
| R1–R4 | Cube B operands | Cube B (R1–R2) | VEC-4K-v2 / MTE — free |
| R5–R7 | Vector / MTE | Vector / MTE | Vector / MTE — free |
| W0 | Cube C drain | Cube C drain | VEC-4K-v2 / MTE — free |
| W1–W7 | Vector / MTE | Vector / MTE | Vector / MTE — free |

VEC-4K-v2 binding: **R0 (Port A, with `is_xpose_A`)**, **R4 (Port B, with `is_xpose_B`)**, **W0 (D0)**, **W4 (D1)**. Mask `C` rides on whichever value-tile read port is idle (1–2 strips per fetch).

---

## 10. Out-of-Order Execution Model — Foundations

> **(v1 → v2: 本章基础部分完整复制自 v1 §10。v2 增量为 Tile Metadata RAT(§10.3 引用)与 §10.6 指向 §11 的扩展投机恢复机制。)**

The Davinci-v2 core implements a **ROB-less out-of-order** execution model. Because the core does not need to maintain precise architectural state (no interrupts, no exceptions in run-to-completion kernels), it dispenses with the Reorder Buffer entirely. This section describes how instructions flow through the core and how correctness is maintained.

### 10.1 Core Principles (v1 §10.1, 未变更; v2 增加第 6 条)

1. **OoO dispatch, OoO execution, OoO completion.** An instruction's result is committed to the physical register file as soon as execution completes. There is no in-order retirement stage.
2. **False dependencies (WAW, WAR) eliminated by register renaming.** Both the Scalar RAT (32→128) and Tile RAT (32→256) — and the new v2 **Tile Metadata RAT** — assign each destination to a unique physical register/tile, so no instruction ever overwrites another's live data.
3. **True dependencies (RAW) resolved by tag-based wakeup.** Scalar instructions wait for source tags on the CDB. Tile-domain instructions wait for Tile RAT ready bits signaling physical tile completion.
4. **Branch recovery via RAT checkpoints.** On mispredict, both the Scalar RAT and Tile RAT are flash-restored in 1 cycle; all younger instructions are flushed.
5. **Physical registers freed by reference counting.** No ROB means no retirement-based freeing; instead, a register (scalar or tile) is freed when it is both *orphaned* (no longer the current mapping for any architectural register) and its reference count reaches zero.
6. **(v2 增量) Speculative memory side effects gated by branch tag.** Speculative scalar stores live in the SSB and speculative bulk tile stores live in the STQ until their `branch_tag` resolves to non-speculative. See §11.

### 10.2 Instruction Lifecycle (v1 §10.2, 未变更)

```
  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
  │Fetch│──▶│Decode│──▶│Rename│──▶│ Disp│──▶│Issue│──▶│ Exec│──▶│  WB │
  │ F1-2│   │  D1  │   │  D2  │   │ DS  │   │ IS  │   │EX1-n│   │     │
  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘
                         │                                         │
                    Allocate P-reg                          Write P-reg
                    Update RAT                             Broadcast CDB
                    Checkpoint (branch)                    Wakeup dependents
                    Increment ref-counts                   Decrement ref-counts
                                                           Free orphans (refcnt=0)
```

Detailed per-stage actions (v1 §10.2 + v2 增量 in italics):

| Stage | Actions |
|-------|---------|
| **Fetch (F1–F2)** | PC → I-cache + branch predictor; receive 4 instructions |
| **Decode (D1)** | Decode opcode, identify domain, extract fields |
| **Rename (D2)** | Read Scalar RAT / Tile RAT for sources (get physical tags + ready bits); allocate physical dest from appropriate free list; update RAT; increment ref-counts for source physical regs/tiles; if branch: allocate + flash-copy both RAT checkpoints. *(v2: also stamp branch_tag on µop; if branch, allocate branch_tag from 8-slot pool, snapshot SSB head + STQ head into checkpoint)*. |
| **Dispatch (DS)** | Write entry into target RS with opcode, source tags, ready bits, dest tag. *(v2: include branch_tag; allocate SSB slot for stores, STQ slot for bulk tile stores)*. |
| **Issue (IS)** | Select oldest ready entry; read physical RF for operands not yet captured |
| **Execute (EX)** | Compute result (variable latency per unit). *(v2: scalar stores deposit into SSB slot, MTE bulk stores deposit address+data-ptr into STQ slot — no L1-D / memory write yet)*. |
| **Writeback (WB)** | Scalar: broadcast (tag, data) on CDB; write result to physical RF; set ready bit in Scalar RAT; wakeup dependent RS entries. Tile: write data to physical tile in TRegFile-4K; set ready bit in Tile RAT; wakeup dependent tile RS entries. Both: decrement ref-counts for source regs/tiles; if source is orphan and refcnt=0: free it. *(v2: SSB / STQ entries become drain-ready when their branch_tag clears; data/tile actually commits to L1-D / external memory only on drain.)* |

### 10.3 Register Alias Table (RAT) Operation

> **(v1 → v2: 复制自 v1 §10.3。v2 在 Tile RAT 旁增加了 Tile Metadata RAT,语义见 §6.1。)**

The two RATs (Scalar and Tile) plus the new Tile Metadata RAT are the core state machines of the processor. In the absence of a ROB, the RATs are the **definitive mapping** from architectural to physical registers.

**Scalar RAT rename example (4-wide, single cycle, v1, 未变更):**

```
  Instruction stream:
    i0:  ADD  X5, X2, X3
    i1:  MUL  X6, X5, X7
    i2:  SUB  X5, X8, X9
    i3:  ADD  X10, X5, X6

  Before rename:
    Scalar RAT: X2→P2, X3→P3, X5→P5, X7→P7, X8→P8, X9→P9

  Rename (all in one cycle at D2):
    i0: src1=P2, src2=P3, dst=P40 (new);  RAT: X5→P40;  P5 marked orphan
    i1: src1=P40 (bypass from i0), src2=P7, dst=P41;  RAT: X6→P41
    i2: src1=P8, src2=P9, dst=P42 (new);  RAT: X5→P42;  P40 marked orphan
    i3: src1=P42 (bypass from i2), src2=P41 (bypass from i1), dst=P43;  RAT: X10→P43

  After rename:
    Scalar RAT: X2→P2, X3→P3, X5→P42, X6→P41, X7→P7, X8→P8, X9→P9, X10→P43
    Orphans: P5 (old X5), P40 (old X5, transient within group)
    Free when: orphan AND refcount=0
```

**Tile RAT rename example (v1, 未变更):**

```
  Instruction stream:
    i0:  TILE.LD  T10, [X5]            (scalar src X5, tile dst T10)
    i1:  TILE.LD  T20, [X6]            (scalar src X6, tile dst T20)
    i2:  VADD     T10, T10, T20        (tile src T10, T20; tile dst T10)
    i3:  TILE.ST  [X7], T10            (scalar src X7, tile src T10)

  Before rename:
    Tile RAT: T10→PT10, T20→PT20

  Rename (all in one cycle at D2):
    i0: scalar src=<from Scalar RAT>, tile dst=PT100 (new);  Tile RAT: T10→PT100;  PT10 orphaned
    i1: scalar src=<from Scalar RAT>, tile dst=PT101 (new);  Tile RAT: T20→PT101;  PT20 orphaned
    i2: tile src1=PT100 (bypass i0), tile src2=PT101 (bypass i1),
        tile dst=PT102 (new);  Tile RAT: T10→PT102;  PT100 orphaned
    i3: tile src=PT102 (bypass i2);  no tile dst (store)

  After rename:
    Tile RAT: T10→PT102, T20→PT101
    Tile orphans: PT10 (old T10), PT20 (old T20), PT100 (old T10, transient)
    Free when: orphan AND tile refcount=0
```

**Tile Metadata RAT (v2 增量):** A 256 × 32 b sibling SRAM holding `(shape.x, shape.y, format)` per physical tile. Updated by retire-time writes from producing instructions; read together with the tile's first strip during operand fetch (§9.2.1).

### 10.4 Common Data Bus / Tile Completion Bus

> **(v1 → v2: 端口数 / 数据线宽完整复制自 v1 §10.4。v2 在 TCB 每端口增加 1 b "metadata committed" 标志。)**

The CDB is a broadcast network connecting execution unit outputs to reservation stations and the physical register file.

| Parameter | Value |
|-----------|-------|
| CDB ports | **6** (4 ALU + 1 MUL/LSU + 1 TILE.GET) |
| Broadcast width | 7-bit tag + 64-bit data per port |
| Snoop points | All RS entries (32+24+24+4+16 = 100 entries × 2 scalar sources in v2) |

The CDB carries scalar physical register tags and 64-bit data. Instructions that produce tile results use the **Tile Completion Bus (TCB)** instead. The CDB is used by:
- Scalar ALU, MUL/DIV, Branch, LSU results (5 ports)
- **TILE.GET** that produces a scalar GPR result (shared port 6)

**Tile Completion Bus (TCB):** A lightweight broadcast network (8-bit physical tile tag, no data payload, **+1 b metadata-committed in v2**) with **4 ports** supporting up to 4 simultaneous tile completions per cycle. TCB port allocation:

| TCB port | Source |
|----------|--------|
| TCB0 | Vector unit (VEC-4K-v2 ALU/FMA/PTO destination tile) |
| TCB1 | Cube unit (CUBE.DRAIN destination tile) |
| TCB2 | MTE unit — TILE.LD / TILE.GATHER / TILE.ZERO / TILE.COPY / TILE.PUT completion (port 1) |
| TCB3 | MTE unit — TILE.LD / TILE.GATHER / TILE.ZERO / TILE.COPY / TILE.PUT completion (port 2) |

Each tile-domain RS entry compares its physical tile source tags against all 4 TCB tags for wakeup, mirroring CDB wakeup for scalar RS entries. MTE instructions that do not produce a tile destination (TILE.ST, TILE.SCATTER) do not broadcast on the TCB.

Each CDB port can broadcast one result per cycle. When a result is broadcast (v1 §10.4, 未变更):

1. **Wakeup:** Every scalar/LSU RS entry compares its source tags against the broadcast tag. On match, the ready bit is set and data is captured.
2. **RF write:** The physical register file captures the data at the destination tag.
3. **RAT status:** The ready bit for the physical register is set in the RAT status table.

### 10.5 Physical Register Freeing (Reference Counting)

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §10.5。)**

Without a ROB, the processor cannot use retirement to determine when a physical register is dead. Instead, it uses a **reference counting** scheme, applied identically to both scalar physical registers and physical tile registers:

```
  Per physical scalar register (128 entries):
    ┌──────────┬──────────┬───────────┐
    │ orphan   │ refcount │ state     │
    │ (1 bit)  │ (4 bits) │           │
    └──────────┴──────────┴───────────┘

  Per physical tile register (256 entries):
    ┌──────────┬──────────┬───────────┐
    │ orphan   │ refcount │ state     │
    │ (1 bit)  │ (3 bits) │           │
    └──────────┴──────────┴───────────┘

  State machine (same for both):
    MAPPED:   RAT points to this register; refcount tracks in-flight readers
    ORPHAN:   RAT no longer points here (remapped); refcount may be > 0
    FREE:     orphan AND refcount == 0 → returned to free list
```

Tile refcount is 3 bits (max 7 concurrent readers per physical tile), which suffices because the Vector RS (24 in v2), Cube RS (4), and MTE RS (16) issue at most a few readers per tile simultaneously. Scalar refcount is 4 bits (max 15).

**Lifecycle events (identical for scalar and tile):**

| Event | orphan | refcount | Action |
|-------|--------|----------|--------|
| Allocated as destination at D2 | 0 | 0 | Added to RAT mapping |
| Instruction reads this register (dispatched to RS) | — | +1 | Reader registered |
| Reader completes execution (reads at IS/EX) | — | −1 | Reader done |
| RAT remaps arch-reg to new physical register | 1 | — | Old mapping becomes orphan |
| refcount reaches 0 while orphan=1 | 1 | 0 | **Free**: return to free list |

**Branch misprediction and ref-counts:** When a mispredict occurs, all instructions younger than the branch are flushed. Their RS entries are invalidated, and the ref-counts for their source registers (scalar and tile) are decremented. Physical registers/tiles allocated as destinations by flushed instructions are returned directly to their respective free lists. Both free list head pointers are restored from the checkpoint to reclaim all speculatively allocated registers.

### 10.6 Branch Recovery

> **(v1 → v2: v1 的 1-cycle dual-RAT restore 序列完整保留;v2 在并行加上 SSB/STQ flush。详细投机恢复机制见 §11。)**

```
  ┌────────────────────────────────────────────────────────────┐
  │  Branch Misprediction Recovery (v1 baseline + v2 增量)     │
  │                                                            │
  │  Cycle 0: Branch resolves as MISPREDICTED at EX1          │
  │    → Identify checkpoint ID and branch_tag from RS entry  │
  │                                                            │
  │  Cycle 1: Recovery actions (all in parallel):              │
  │    a) Flash-restore: checkpoint[id] → active Scalar RAT   │
  │    b) Flash-restore: checkpoint[id] → active Tile RAT     │
  │    c) Restore scalar free-list head pointer                │
  │    d) Restore tile free-list head pointer                  │
  │    e) Restore RAS top pointer from checkpoint              │
  │    f) Invalidate all RS entries younger than branch         │
  │    g) Decrement ref-counts (scalar + tile) for flushed ops │
  │    h) Deallocate all checkpoints younger than this branch  │
  │    i) (v2) Flush SSB entries with btag = mispredict_tag    │
  │       or any descendant tag (parallel CAM clear, §11.4.4)  │
  │    j) (v2) Flush STQ entries similarly (§11.5.1)           │
  │    k) (v2) Reset Tile-Metadata RAT delta-tag from checkpt  │
  │                                                            │
  │  Cycle 2: Redirect fetch PC to correct branch target       │
  │                                                            │
  │  Cycle 3+: New instructions begin entering F1               │
  │                                                            │
  │  Total penalty: 6 cy (v1) → 6 cy (v2; SSB/STQ flush is     │
  │                       parallel and adds 0 cy hot-path)     │
  └────────────────────────────────────────────────────────────┘
```

The basic v1 branch-recovery sequence (RAT flash-restore, free-list-head restore, RS flush) is preserved. **Section 11 specifies the additional mechanisms required for safe speculative execution beyond branch prediction**: SSB / STQ flush, branch-tag ancestry tracking, and the proof that this set of mechanisms is **sufficient to keep architectural state correct without a Reorder Buffer**.

---

## 11. Speculative Execution Recovery Without a Reorder Buffer

> **Question:** The v1 design eliminates the Reorder Buffer (ROB) by leveraging the no-precise-exception envelope, using the Reservation Station + reference-counting + RAT-checkpoint trio for OoO execution. v2 adds branch-prediction-driven **speculative execution** to extend the OoO window past unresolved branches. Can we do this safely — i.e., guarantee that a misspeculated path **never** corrupts architectural state — **without** introducing a ROB?
>
> **Answer: Yes, with two additional structures (Speculative Store Buffer and Speculative Tile-Store Queue) plus a small Branch-Tag Ancestry Tracker. This section proves it and details the mechanisms.**

### 11.1 What the ROB traditionally does

Textbook OoO processors use a Reorder Buffer to provide three services bundled together:

1. **Precise architectural state** — every instruction completes (writes to architecturally-visible state) **in program order**. On exception, the ROB lets the processor identify the exact instruction that faulted and discards everything younger.
2. **Speculative recovery for memory-side effects** — stores remain in the ROB (or a coupled store queue) until they retire in order; mispredicted-path stores are simply not retired.
3. **Resource freeing in program order** — physical registers / tiles are returned to the free list when the consuming arch-reg's old mapping retires.

The Davinci-v1 design noted that **service (1) is unneeded** in the run-to-completion AI-kernel envelope, that **service (3) is replaced by reference counting**, and that **service (2) does not arise** because the processor doesn't speculate past an unresolved branch (it instead flushes the pipeline at every mispredict — only flushing **non-stored** state, since v1 stores commit to L1-D OoO once the producing branch has resolved).

The v1 caveat — *"don't speculate past an unresolved branch"* — is restrictive: it limits the OoO window to the time between branches, which is typically only 5–10 instructions in tight scalar loops. v2 lifts this restriction by allowing instructions younger than an unresolved branch to dispatch, issue, and execute speculatively. **Doing this correctly requires service (2) to be re-introduced — but only service (2), not (1) or (3).**

### 11.2 Categorizing speculative state

When an instruction executes speculatively (i.e. depends on an unresolved branch), its effects fall into one of three classes:

| Class | What it touches | Who recovers it on flush | Currently handled by |
|-------|-----------------|---------------------------|----------------------|
| **A. Renamed register / tile state** | Writes to a physical scalar register (P0–P127), a physical tile (PT0–PT255), or a metadata RAT entry | Returns to free list once orphan + refcount=0 (no ROB needed) | RAT checkpoint flash-restore + refcount + free-list-head restore |
| **B. Pipeline state** | Occupies an RS slot; in flight in EX stages; consumes CDB/TCB cycles | Branch-tag CAM-clear invalidates the RS entry; in-flight EX is flushed at WB | Branch-tag stamping at D2 (§5.1, §6.2) |
| **C. Externally-visible state** | Writes to L1-D / L2 cache, scatters to memory, MMIO accesses, fences/barriers, cross-core observable ordering | **Cannot be recovered** once it leaves the core | **NEW: SSB (§11.4) and STQ (§11.5) gate these to never *reach* memory until non-speculative** |

Class A is fully handled by the v1 mechanisms — the RAT flash-restore + refcount path is **logically equivalent** to a ROB's "retire on commit" for register/tile state, because:

- A misspeculated destination is simply *never* the architectural mapping (RAT restore overwrites it).
- A misspeculated source-register read either consumed a valid value (which produces no side effect, just a wasted compute) or stalled until the branch resolved (in which case the RS entry is flushed before the read).

Class B is fully handled by the branch-tag CAM-clear: every RS / reservation-station entry tagged with the mispredicted branch (or any *younger* branch in its dependency chain) is invalidated in one cycle, and the corresponding physical-register/tile destinations roll back to the free list.

**Class C is the only class that needs a new mechanism**, and the new mechanism only needs to ensure that any class-C effect *of a speculative instruction* is **delayed** until the instruction is known to be on the correct path. This is the core insight: **we don't need a ROB to track instruction order globally; we only need to gate the externally-visible side effects by branch tag**.

### 11.3 The Branch-Tag Speculation Tracker

The processor maintains a **branch-tag tracker** — a small structure with three components:

```
  ┌───────────────────────────────────────────────────────────────┐
  │  Speculation Tracker  (always 8 active branch tags max)       │
  │                                                               │
  │  (a) Tag-state vector: 8 × 2 b state                          │
  │       0 = free                                                │
  │       1 = speculative (branch not yet resolved)               │
  │       2 = correct (branch resolved correctly; tag draining)   │
  │       3 = wrong (branch mispredicted; tag flushing)           │
  │                                                               │
  │  (b) Ancestry bitmap: 8 × 8 b symmetric matrix                │
  │       anc[i][j] = 1 iff branch i is an ancestor of branch j   │
  │       (i.e. j was fetched while i was unresolved)             │
  │       Allocated at D2 when each new branch is renamed.        │
  │                                                               │
  │  (c) Instruction-tag map: maintained in RS entries' btag      │
  │       field (3 b each). Already part of every RS entry §7.2.  │
  └───────────────────────────────────────────────────────────────┘
```

#### 11.3.1 Tracker operation

- **Branch enters D2:** allocate a free tag `t`. Set `state[t] = speculative`. Snapshot ancestry: `anc[t][:] = anc[parent_t][:] | (1 << parent_t)` where `parent_t` is the youngest still-speculative branch's tag (or all zeros if none). The branch's RS entry stamps `btag = t`. All instructions following until the next branch also stamp `btag = t`.
- **Branch resolves correctly at EX1:** set `state[t] = correct`. The drain logic propagates this to SSB / STQ entries, which then clear their `btag` (replacing it with the next-older speculative tag, if any). When `btag = 0xFF` (no older speculative branches remain), the entry becomes `drain_rdy`.
- **Branch mispredicts at EX1:** set `state[t] = wrong`. **Atomically** clear all RS / SSB / STQ entries with `btag` matching `t` *or* matching any descendant of `t` (computed from `anc[t][:]`). The RAT flash-restore from checkpoint completes in the same cycle.
- **Tag freed:** when an entry transitions from speculative to non-speculative *and* fully drains (or is invalidated), the corresponding tag's allocation in the tracker is released. A tag is freed when no in-flight entry references it.

#### 11.3.2 Tracker area

| Component | Size | Gate count |
|-----------|------|------------|
| Tag-state vector | 8 × 2 b = 16 FF | ~200 gate |
| Ancestry bitmap | 64 b register | ~700 gate |
| State-update FSM (allocate, resolve, mispredict) | ~3 K gate |
| Drain-broadcast wiring (to SSB, STQ, vector RS, etc.) | ~1 K gate |
| **Total** | | **~5 K gate** (negligible — ~0.001 mm² @ 5 nm) |

### 11.4 Speculative Store Buffer (SSB) — Scalar Memory Path

The SSB is the **gate** between the LSU's store pipeline and the L1-D cache. Already introduced in §8.2.1; this subsection details the speculation-recovery mechanism it enables.

#### 11.4.1 Allocation and population

```
  D2 (rename) of a scalar store instruction:
    Allocate SSB slot k from the free pool (FIFO order)
    Set SSB[k] = {valid=1, btag=current_btag, addr=⊥, data=⊥, drain_rdy=0}
    Stamp the LSU-RS entry with ssb_idx = k

  IS / EX (issue + execute):
    LSU computes addr and reads data from physical RF (or captures from CDB)
    SSB[k].addr ← addr
    SSB[k].data ← data
    SSB[k].size ← size
    LSU-RS entry retires (released; no CDB broadcast for stores)
```

At this point the store's address and data are **fully resolved**, but the store has **not** committed to L1-D. The store's effect on memory is **isolated within the SSB**.

#### 11.4.2 Drain to L1-D (when non-speculative)

```
  When tag t becomes correct:
    For each SSB[k] with valid=1 && btag=t:
      btag_new = next-older speculative tag in this entry's history (from §11.3 ancestry)
      if btag_new == 0xFF (no older speculation): SSB[k].drain_rdy ← 1
      else: SSB[k].btag ← btag_new (still speculative)

  Drain pump (1 store per cycle):
    Pick oldest SSB[k] with valid && drain_rdy
    Issue write to L1-D pipeline (same path as v1 store commit)
    On L1-D ack: SSB[k].valid ← 0, slot returned to free pool
```

#### 11.4.3 Forwarding to loads

Loads can forward from the SSB on address match, with the **branch-ancestry constraint**:

```
  For a load (addr, btag_load):
    For each SSB entry e with e.valid && addr_match(e.addr, addr) && size_compatible(e):
      if e.btag == 0xFF:                      # store is non-speculative — always OK
         forward
      elif anc[e.btag][btag_load]:           # store is speculative on load's ancestry chain
         forward
      else:                                   # store is on a different speculation path
         skip (don't forward; load goes to L1-D, where it sees the pre-mispredict view)
```

Address-ambiguous stores (older stores still computing addresses) cause the load to wait, as in v1.

#### 11.4.4 Mispredict invalidation

```
  When tag t becomes wrong (mispredict at EX1):
    descendants = {j : anc[t][j] = 1} ∪ {t}
    
    For each SSB[k] with valid && btag ∈ descendants:
      SSB[k].valid ← 0   (entry invalidated; never reaches L1-D)
    SSB free-list head pointer ← restored from checkpoint[t]
```

Critically, **invalidated stores are silently discarded** — no memory write was issued. Memory state is unaffected by the misspeculated path.

#### 11.4.5 Capacity sizing

- **Min capacity:** at peak, all 8 active speculation tags can have stores in flight. Each tag's "store density" is bounded by the speculation window between branches (typically 5–10 instructions, of which 20–30% are stores → ~2 stores per tag per branch).
- **Sized for:** 8 tags × ~3 stores/tag = 24 entries, matching the v2 SSB capacity.
- **Stall behavior:** if the SSB fills, dispatch stalls at the next store. This is rare in practice (97th-percentile occupancy is ~12 entries on typical kernels) but the mechanism is correct under any occupancy — the front-end simply waits for an SSB slot to drain.

### 11.5 Speculative Tile-Store Queue (STQ) — MTE Memory Path

The STQ is the analogue of the SSB for **bulk tile stores** issued by the MTE unit (`TILE.ST`, `TILE.SCATTER`). Already introduced in §8.5.2; this subsection emphasizes the speculation-recovery semantics.

Key differences from SSB:

- **Data does not reside in the STQ.** The 4 KB tile payload stays in TRegFile-4K, referenced by `tile_phys_idx`. The STQ holds only the *intent* (address, source phys-tile, branch tag).
- **Smaller capacity (8 entries).** Bulk tile stores are infrequent compared to scalar stores.
- **Drain triggers a memory-bound transfer (8-cy TRegFile read epoch + ~64 cy memory write).** Unlike a single-cycle SSB drain, STQ drains take dozens of cycles and overlap with subsequent MTE operations.

#### 11.5.1 Allocation, drain, invalidation

The flow mirrors §11.4.1–§11.4.4 with these adaptations:

```
  Allocation: D2 of TILE.ST or TILE.SCATTER → STQ slot
  Population: at MTE issue, fields {base_addr, tile_phys_idx, stride or scatter_idx_phys} fill in
  Drain: when btag becomes 0xFF, drain_rdy ← 1; oldest drain_rdy entry begins
         streaming the tile from TRegFile (8-cy read epoch) to memory (64-cy write)
  Invalidation: on tag becoming wrong, matching STQ entries set valid ← 0
                The source physical tile is freed via the normal Tile RAT refcount path
                (the entry's allocation incremented refcount; invalidation decrements it)
```

#### 11.5.2 Why 8 entries

- **Min capacity:** 1 STQ entry per active speculation tag = 8.
- **Worst case:** A kernel with 1 TILE.ST per ~10 instructions issues ~1 STQ entry per ~50 ns at 1.5 GHz; the drain rate is ~1 entry per ~72 cy = ~50 ns. The STQ stays at ~1–2 entries average.
- **Stall behavior:** STQ-full is rare; when it occurs, dispatch stalls at the next bulk tile store.

### 11.6 What about other externally-visible operations?

The full taxonomy of externally-visible side effects in the Davinci-v2 ISA is:

| Operation | External effect | Speculation-safe via |
|-----------|-----------------|----------------------|
| Scalar store (SB/SH/SW/SD) | Write to L1-D / L2 / DRAM | **SSB** (§11.4) |
| TILE.ST | 4 KB write to memory | **STQ** (§11.5) |
| TILE.SCATTER | Indexed memory write | **STQ** (§11.5) |
| Scalar load | Reads L1-D, fills physical register; *no external state change* | Recovered via RAT/refcount; load has no externally-visible effect on commit (other than caching, which is not architectural) |
| TILE.LD, TILE.GATHER | Reads memory into TRegFile-4K; *no external state change* | Recovered via Tile RAT/refcount |
| FENCE | Memory ordering barrier | Allocated at D2; **does not execute until btag = 0xFF**. On mispredict, flushed like any other RS entry. |
| AMO atomics (future) | Read-modify-write to memory | Would need to allocate a SSB-like slot held until non-speculative; fits naturally in the SSB. |
| Branch resolve | Updates predictor tables | Predictor update is **conditional on branch correctness**: predictor write fires only when the branch tag becomes "correct" (i.e., on the same drain trigger as SSB/STQ). |
| TCB / CDB broadcast | Wakes dependent RS entries | Internal to core; not externally visible. Mispredicted broadcasts are absorbed by the RS branch-tag CAM-clear. |
| Performance-counter increment | Updates a CSR | Counters are explicitly architectural; v2 reuses the FENCE-style "execute-only-when-non-speculative" gate (§11.6.1). |
| MMIO load/store (future) | I/O side effect | Would need explicit speculation barrier; recommended to gate at the SSB level with a "MMIO" qualifier that forces in-order completion. |

#### 11.6.1 FENCE and CSR semantics under speculation

```
  D2 (rename) of FENCE / CSRRW:
    Allocate RS entry as usual; stamp btag
  
  IS (issue):
    The instruction is held in RS until btag == 0xFF (i.e. no older
    speculation). This is a small extension to the issue-ready
    predicate: in addition to "all source operands ready", we add
    "btag is non-speculative".
  
  EX:
    Execute as normal. By construction, the instruction is on the
    correct path.
```

This adds latency to FENCE / CSR ops (they wait for older branches to resolve) but is correct. In practice, FENCE is rare in tight kernels.

#### 11.6.2 Cache-line state changes (loads)

Speculative loads can pull cache lines into L1-D / L2 that wouldn't have been pulled on the correct path. This is a **microarchitectural** effect (cache-state pollution), not an architectural one — it doesn't violate program semantics. Modern OoO processors all have this property; the well-known Spectre-class side-channel concerns apply equally to v1 (speculative loads were already permitted because they only fill physical registers, not memory). Recovery against side-channel leakage is **out of scope** for this document; standard mitigations (cache partitioning, speculation barriers) are orthogonal to the no-ROB recovery scheme.

### 11.7 What this scheme does NOT provide (and why that's OK)

The branch-tag + SSB + STQ scheme guarantees: **no misspeculated instruction's externally-visible effect ever reaches memory or any architecturally-visible state, regardless of how deep the speculation goes (up to 8 levels) or how many instructions execute speculatively.**

It does **not** provide:

1. **Precise exceptions.** If a load page-faults (in a hypothetical paged Davinci variant), the processor cannot identify the program-order point of the fault — only the renamed-register view. This is the same v1 limitation. The kernel-execution envelope assumes faults don't happen mid-kernel; any fault is treated as a fatal error.
2. **Precise hardware breakpoints / single-step.** Without ordered retirement, hitting a breakpoint at instruction *i* may have already executed a few instructions past *i*. Debug support is degraded but not broken: breakpoints fire at the granularity of the ROB-less retirement cluster (~10-instruction window).
3. **Strict in-order memory ordering for I/O semantics.** Loads and stores still drain to L1-D in store-buffer-FIFO order, so total-store-order (TSO) within a single thread is preserved; but if the core were extended to support SC (sequential consistency) across threads, additional ordering machinery would be needed.
4. **Recovery from non-branch misspeculation.** Memory-disambiguation misspeculations (where a younger load reads from L1-D before an older store's address resolves, then the older store's address turns out to alias) are not handled — the LSU's address-disambiguation logic still requires older stores' addresses to be known before the load issues, as in v1. **Memory dependence prediction is not implemented.** Removing this restriction is an orthogonal extension that would require an additional tag for "load misspeculation" similar to the branch tag.

These limitations are consistent with the v1 design envelope (run-to-completion AI kernel).

### 11.8 Comparison with a ROB-based design

| Aspect | ROB-based design | Davinci-v2 (no ROB) |
|--------|------------------|---------------------|
| Recovery for register / tile state | ROB walks back, undoes mappings in order | RAT flash-restore + refcount free; **same outcome, simpler hardware** |
| Recovery for memory stores | Stores in ROB / coupled SQ; release on retire | **SSB / STQ** with branch-tag gating |
| Recovery for fences / CSR | ROB serializes; instruction retires in order | Issue gated on `btag = 0xFF`; correct but adds 0–6 cy latency |
| Speculative depth | bounded by ROB capacity (typically 64–256 entries) | bounded by **8 active branch tags** (≈ 8-deep nested branches) |
| Storage overhead | ROB ≈ 256 entries × ~150 b = ~5 KB + retirement logic | Branch-tag tracker (5 K gate) + SSB extension (24 vs 16, +8 entries × 182 b ≈ ~1.5 KB) + STQ (8 × 110 b ≈ ~1 KB) ≈ **~2.5 KB total** |
| Wakeup logic | RS does dependency tracking; ROB independent | **RS does dependency tracking AND speculation tracking** via branch-tag CAM (∝ RS size, already paid for) |
| Mispredict penalty | ROB walk-back + flush ≈ 5–10 cy | RAT flash-restore + SSB/STQ tag-CAM-clear ≈ **1 cy parallel restore + 6-cy refill = 7 cy** |
| Precise exceptions | yes (free) | no (out of envelope) |
| Single-thread TSO memory ordering | yes | yes (FIFO drain through SSB) |

**The key insight:** in environments where precise exceptions are not required (the AI-kernel envelope), the ROB's three bundled services unbundle naturally. Service (1) is free if you don't need it. Service (3) is replaced by reference counting. Service (2) — **the only remaining service** — is implemented by the SSB + STQ + branch-tag tracker at a fraction of a ROB's cost.

### 11.9 Cycle-by-cycle example: speculative store followed by mispredict

```
  Cycle  Action
  ─────  ──────────────────────────────────────────────────────────────────
   0     Branch B1 enters D2; allocated tag t=3, state[3] = speculative
   1     SD X5, [X8]+0    — younger than B1 — D2 allocates SSB[7] tagged 3
   2     SD X6, [X8]+8    — younger than B1 — D2 allocates SSB[8] tagged 3
   3     SD X7, [X8]+16   — younger than B1 — D2 allocates SSB[9] tagged 3
   4     ... 5 more in-flight instructions ...
   9     B1 reaches EX1: mispredicted!
  10     state[3] = wrong; all RS entries with btag = 3 invalidated;
         SSB[7], SSB[8], SSB[9] set valid ← 0; tile/scalar free-list heads
         restored from checkpoint[3]; RAT restored from checkpoint[3]
  11     Fetch redirected to correct branch target
  12-16  Front-end refill (5 cy)
  17     First correct-path instruction enters EX1 (total mispredict
         penalty: 17 - 10 = 7 cy front-end refill + 1 cy restore = 8 cy)

  Architectural state at cycle 17:
    Memory: NEVER wrote SSB[7..9]. Cache lines unaffected. Correct.
    X5, X6, X7: physical reg mappings rolled back via RAT restore.
    Free lists: include the orphan physical regs allocated in cycles 1–8.
```

The mispredict is recovered in 8 total cycles — one cycle longer than v1's 6-cy penalty for two reasons:

1. **One additional cycle of redirect** because the branch tag's CAM-clear must propagate to all RS entries before the next instruction enters DS. (v1's 6-cy penalty assumed instantaneous RS flush; v2 makes this explicit.)
2. **One additional cycle of front-end refill** in the worst case where the first correctly-fetched instruction's source register depends on a value being restored to the RAT.

In practice, both effects can be hidden through pipelining (the CAM-clear is parallel; the front-end refill is identical to v1). The realistic mispredict penalty for v2 is **6–7 cycles**, well within the v1 envelope.

### 11.10 Hardware cost summary

| Block | v2 addition | Gate count |
|-------|-------------|------------|
| Branch-tag tracker (state vector + ancestry bitmap + FSM) | new | ~5 K |
| Speculative Store Buffer (24 entries, 182 b each) | extends v1's 16-entry store buffer | ~80 K (incl. CAM) |
| Speculative Tile-Store Queue (8 entries, 110 b each) | new | ~12 K |
| Branch-tag stamping in RS entries (24 + 16 + 24 + 4 + 16 = 84 entries × 3 b) | extends RS entry width | ~2 K |
| Checkpoint extension (+13 b/slot × 8 slots) | extends checkpoint store | ~1 K |
| Tile Metadata RAT (256 × 32 b SRAM, 4R/2W) | new (also serves §6.1) | ~10 K |
| **Total v2 speculation hardware** | | **~110 K gate (~0.025 mm² @ 5 nm)** |

This is **~3.5%** of the ~3.26 mm² total core area (v1) — far less than a comparable-capacity ROB (typically 256 entries × ~10 b/entry with full retirement bypass = ~50 K gate just for storage, plus equivalent forwarding logic that would push a ROB-based v2 to ~300–500 K gate of new structure).

---

## 12. Memory Subsystem

> **(v1 → v2: §12.1 / §12.3 / §12.4 / §12.5 完整复制自 v1 §11。v2 增量集中在 §12.2 Store path,把 v1 的 16-entry store buffer 升级为 24-entry SSB,并加入 STQ。)**

The memory subsystem is structurally identical to v1 §11. The two changes are integration points for the SSB and STQ.

### 12.1 Cache Hierarchy (v1 §11.1, 未变更)

```
  ┌────────────┐    ┌────────────┐
  │  L1-I      │    │  L1-D      │
  │  64 KB     │    │  64 KB     │
  │  4-way     │    │  4-way     │
  │  2-cy lat  │    │  4-cy lat  │
  └─────┬──────┘    └─────┬──────┘
        │                 │
        └────────┬────────┘
                 ▼
        ┌────────────────┐
        │  L2 (Unified)  │
        │  512 KB        │
        │  8-way         │
        │  12-cy lat     │
        └───────┬────────┘
                │
                ▼
        External Bus / NoC
```

| Cache | Size | Associativity | Line size | Latency | Ports | MSHRs |
|-------|------|---------------|-----------|---------|-------|-------|
| L1-I | **64 KB** | 4-way | 64 B | **2** cycles | 1 read (fetch) | 4 |
| L1-D | **64 KB** | 4-way | 64 B | **4** cycles | 1 read + 1 write (LSU) | 8 |
| L2 | **512 KB** | 8-way | 64 B | **12** cycles | 1 read + 1 write | 16 |

### 12.2 Store Path (v2 增量)

```
  Scalar store:      LSU-RS → SSB (24 entries) → L1-D (only on tag-clear)
  Bulk tile store:   MTE-RS → STQ (8 entries)  → MTE memory pipeline → L2 (only on tag-clear)
```

The L1-D's existing 8 MSHRs and 4-cy store pipeline are unchanged. The SSB inserts in front of L1-D as a CAM-addressable forwarding buffer; it already played that role in v1's 16-entry store buffer, so the L1-D interface is unchanged. v2 widens the buffer to 24 entries and adds branch-tag gating (full design in §11.4).

| Property | v1 store buffer | v2 SSB |
|----------|-----------------|--------|
| Entries | 16 | **24** |
| Forwarding | yes | yes (now btag-aware §11.4.3) |
| Branch-tag gating | — | **yes (§11.4)** |
| Drain to L1-D | OoO upon resolve | only when btag = `0xFF` |

### 12.3 TLBs (v1 §11.2, 未变更)

| TLB | Entries | Associativity | Page sizes | Miss penalty |
|-----|---------|---------------|------------|-------------|
| I-TLB | **64** | Fully assoc | 4 KB, 2 MB | L2 TLB lookup |
| D-TLB | **64** | Fully assoc | 4 KB, 2 MB | L2 TLB lookup |
| L2 TLB (unified) | **512** | 8-way | 4 KB, 2 MB, 1 GB | Page table walk |

### 12.4 MTE Memory Path (v1 §11.4, 未变更)

The MTE unit has a **high-bandwidth path** to the L2 cache (and external memory) for tile data transfers, separate from the scalar LSU path through L1-D.

```
  MTE ──▶ L2 Cache (512 KB) ──▶ External Memory
           64 B/cy sustained bandwidth
           1 cache line per cycle
           1 tile (4 KB) = 64 cache lines = 64 cycles from L2
```

| Parameter | Value |
|-----------|-------|
| MTE → L2 bandwidth | **64 B/cycle** (1 cache line/cycle) |
| Tile load from L2 (hit) | **64 cycles** per tile (4 KB / 64 B) |
| Tile load from external memory | **200–400 cycles** per tile (DRAM dependent) |
| Outstanding MTE requests | **32** (deep buffer for memory-level parallelism) |
| Prefetch support | MTE RS can issue TILE.LD early, buffering data in TRegFile |

The MTE unit exploits the large TRegFile-4K (256 tiles, 1 MB) as a **software-managed scratchpad**. Programmers (or compiler) schedule TILE.LD instructions well ahead of CUBE.OPA to hide memory latency. The 32-entry outstanding request buffer allows many tile loads to be in flight simultaneously, maximizing bandwidth utilization.

**v2 增量:** `TILE.ST` and `TILE.SCATTER` traffic on this path is gated by the 8-entry STQ (§11.5).

### 12.5 Memory Ordering (v1 §11.5, 未变更; v2 增加 SSB 备注)

Within a single thread:

- **Scalar loads and stores** maintain **program order** through the LSU's address disambiguation (store-to-load forwarding, load queue snooping). In v2 the forwarding source is the SSB (24 entries) instead of v1's 16-entry store buffer; semantics identical.
- **TILE.LD/ST** operations are **unordered** with respect to each other by default. Software uses `FENCE` instructions when ordering between tile operations and scalar operations is required.
- **CUBE.OPA** reads from TRegFile-4K are ordered with respect to preceding TILE.LD operations by the **Tile RAT ready bits** (the cube RS will not issue until the source physical tiles are marked "ready" by completed TILE.LD operations).

The SSB's branch-tag gating is **orthogonal** to memory ordering: stores still drain in alloc-age order, just only when their branch tag is non-speculative (§11.4.2).

---

## 13. Mixed-Domain Instruction Scheduling

> **(v1 → v2: 子节 13.A / 13.B / 13.C / 13.D 完整复制自 v1 §12.1 / §12.2 / §12.3 / §12.4。v2 增量为 §13.1 / §13.2 / §13.3。)**

All four domains share the same front-end, dispatch to domain-specific RSs, and synchronize through Tile RAT ready bits / TCB / CDB.

### 13.A Unified Front-End, Distributed Back-End (v1 §12.1, 未变更)

All four instruction domains share the same front-end pipeline (fetch, decode, rename). At dispatch, instructions are routed to domain-specific reservation stations. This allows the core to exploit instruction-level parallelism across domains:

```
  Single instruction stream (architectural tile regs T0–T31):
    ADD   X5, X2, X3        → Scalar RS → ALU
    TILE.LD T10, [X5]       → Tile RAT: T10→PT200;  MTE RS (ptdst=PT200, depends on X5 via CDB)
    TILE.LD T20, [X6]       → Tile RAT: T20→PT201;  MTE RS (ptdst=PT201, independent)
    VADD  T30, T10, T20     → Tile RAT: T30→PT202;  Vector RS (ptsrc=PT200,PT201; depends via TCB)
    CUBE.OPA z0, T10, T20, r1  → Cube RS (ptsrc=PT200,PT201; depends via TCB ready bits)
    TILE.GET X7, T30, X8    → MTE RS (ptsrc=PT202, depends via TCB; pdst=P60 → CDB scalar result)
    TILE.PUT T10, X9, X10   → Tile RAT: T10→PT203; MTE RS (ptsrc=PT200_old, ptdst=PT203; RMW)
    ADD   X11, X7, X9       → Scalar RS → ALU (depends on X7 via CDB from TILE.GET)
```

### 13.B Cross-Domain Dependencies (v1 §12.2, 未变更; v2 增加投机条目见 §13.3)

Dependencies between domains are tracked through shared mechanisms:

| Dependency | Mechanism |
|------------|-----------|
| **Scalar → MTE** (address operands) | MTE RS entry holds scalar P-reg tag for base address; wakeup via CDB when scalar ALU produces address |
| **Scalar → Vector** (scalar operand in vector reduction) | Vector RS entry holds scalar P-reg tag for scalar inputs; wakeup via CDB |
| **MTE → Vector** (tile data readiness) | Tile RAT: TILE.LD completes → sets ready bit for physical tile; Vector RS wakes via TCB |
| **MTE → Cube** (tile data readiness) | Tile RAT: TILE.LD completes → sets ready bit for physical tile; Cube RS wakes via TCB |
| **Vector → Cube/MTE** (vector result tile) | Tile RAT: vector write completes → sets ready bit; downstream RS entries wake via TCB |
| **Cube → MTE** (drain result tile) | Tile RAT: CUBE.DRAIN completes → sets ready bit for physical tile; MTE RS wakes via TCB |
| **Tile → Scalar** (TILE.GET element extract) | TILE.GET reads physical tile, extracts element, broadcasts scalar result on CDB |
| **Scalar → Tile** (TILE.PUT element insert) | TILE.PUT reads scalar GPR via CDB wakeup, reads old physical tile, writes new physical tile; TCB broadcast |
| **Vector → Vector** (reduction result) | VEC reduction ops produce column/row-vector tile result (TCB completion) |

### 13.C Tile RAT Wakeup & Tile Completion Bus (TCB) — (v1 §12.3, 未变更)

The Tile RAT maintains a **ready bit** per physical tile register (256 bits total). This replaces a scoreboard: rename ensures every tile destination gets a unique physical tile, so there are no WAW/WAR hazards. The ready bit simply tracks whether the producing operation has finished writing the physical tile.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Tile RAT Ready Bits + Tile Completion Bus (TCB)                │
  │                                                                 │
  │  Tile RAT: 32 entries (arch T0–T31) → phys PT0–PT255           │
  │  Ready array: 256 bits (one per physical tile)                  │
  │  TCB: 4 broadcast ports (8-bit tag each, no data payload)      │
  │                                                                 │
  │  TILE.LD T10 renamed:    Tile RAT T10→PT200; ready[PT200] ← 0 │
  │  TILE.LD PT200 completed: ready[PT200] ← 1; TCB broadcast PT200│
  │                                                                 │
  │  VADD T30,T10,T20 renamed: T30→PT202, reads PT200,PT201       │
  │    RS entry: ptsrc1=PT200, ptsrc2=PT201, ptdst=PT202           │
  │    TCB snoop: waits for ready[PT200] && ready[PT201]           │
  │  VADD PT202 completed:   ready[PT202] ← 1; TCB broadcast PT202│
  │                                                                 │
  │  CUBE.OPA reads T10→PT200: checks ready[PT200]                │
  │    if 0 → stall in Cube RS (waits for TCB wakeup)              │
  │    if 1 → issue                                                 │
  │                                                                 │
  │  CUBE.DRAIN writes T12→PT205: ready[PT205] ← 0 at rename      │
  │  CUBE.DRAIN completed:  ready[PT205] ← 1; TCB broadcast PT205 │
  │                                                                 │
  │  TILE.ST reads T12→PT205: checks ready[PT205]                 │
  └─────────────────────────────────────────────────────────────────┘

  TCB wakeup logic (per tile-domain RS entry):
    For each RS entry with N tile sources (up to 3 in v2):
      if (ptsrc_k == TCB_tag && !trdy_k):  trdy_k ← 1
    Ready to issue when all trdy bits set (and scalar rdy if applicable)
```

### 13.D Concurrent Execution Example (v1 §12.4, 未变更)

A typical transformer inference kernel mixes all four domains:

```
  Cycle  │ Scalar ALU │ LSU        │ Vector            │ MTE             │ Cube MXU
  ───────┼────────────┼────────────┼───────────────────┼─────────────────┼──────────────
  0–7    │ addr calc  │ scalar LD  │ —                 │ TILE.LD T0-T3   │ —
  8–15   │ loop ctrl  │ scalar LD  │ —                 │ TILE.LD T4-T7   │ —
  16–23  │ addr calc  │ —          │ VADD read epoch   │ TILE.LD T8-T11  │ CUBE.OPA z0,...
  24–31  │ addr calc  │ —          │ VADD write epoch  │ TILE.LD T12-T15 │ (OPA continues)
  32–47  │ addr calc  │ scalar ST  │ VMUL (16cy)       │ TILE.LD (next)  │ (OPA continues)
  48–63  │ loop ctrl  │ —          │ VCVT (16cy)       │ TILE.ST T16     │ CUBE.DRAIN z0
  64+    │ next iter  │ —          │ —                 │ TILE.LD (next)  │ CUBE.OPA z1,...
```

Key observations:
- Scalar ALU computes addresses and loop control concurrently with cube execution.
- MTE loads next tiles while cube processes current tiles (double-buffering at software level).
- Vector unit handles element-wise operations (activation functions, normalization) in parallel.
- All domains proceed independently, limited only by true data dependencies.

---

### 13.1 New scheduling considerations under speculation (v2 增量)

| Scenario | v1 behaviour | v2 behaviour |
|----------|--------------|--------------|
| Speculative TILE.LD | Couldn't be issued past branch | Can issue speculatively; on mispredict, allocated physical tile rolls back via Tile RAT/refcount |
| Speculative VEC op | Couldn't be issued past branch | Can issue speculatively; staging registers, Acc, etc. are renamed-tile-only and rollback via Tile RAT |
| Speculative TILE.ST | Couldn't be issued past branch | Issues speculatively to STQ; STQ-full → dispatch stall |
| Speculative CUBE.OPA | Couldn't be issued past branch | Can issue speculatively; cube unit's accumulator is in physical-tile rename space |
| FENCE under speculation | Held until branch resolves | Held in RS until `btag = 0xFF` (issue-gated) |

### 13.2 Speculative tile-domain ops

A subtle point: vector / cube / MTE-load instructions younger than an unresolved branch can execute speculatively because:

- Their inputs (source physical tiles) are versioned via the Tile RAT — even if the wrong physical tile is read, the instruction simply commits to its (also-renamed) destination physical tile.
- Their destination is a fresh physical tile that gets freed via refcount + free-list-restore on misspeculation.
- They consume TRegFile-4K port cycles (epoch slots) but don't change architectural memory state.

The only "wasted" resource on misspeculation is **TRegFile-4K port bandwidth** and **microcode beats** spent on the wrong path. In a typical kernel where vector ops are 20–30% of dispatch volume, ~95% branch prediction accuracy means ~1–1.5% of vector compute is wasted on misspeculated paths — well within budget.

### 13.3 Cross-domain dependency table (v2 adds two rows)

| Dependency | Mechanism |
|------------|-----------|
| Scalar → MTE (address operands) | MTE RS holds scalar P-reg tag; CDB wakeup |
| Scalar → Vector (scalar operand SX/SY) | Vector RS holds scalar P-reg tag; CDB wakeup; OR captured at issue-time GPR read |
| MTE → Vector / Cube (tile data readiness) | Tile RAT ready bit + TCB wakeup |
| Vector → Cube/MTE | Tile RAT ready bit + TCB |
| Cube → MTE | Tile RAT ready bit + TCB |
| Tile → Scalar (TILE.GET) | CDB scalar broadcast |
| Scalar → Tile (TILE.PUT) | TCB tile broadcast |
| **Branch → Speculative Memory** | **SSB / STQ branch-tag gating (§11.4, §11.5)** |
| **Branch → Speculative Register / Tile** | **RAT checkpoint flash-restore + refcount free-list-head restore (§11.3)** |

---

## 14. Performance Targets

### 14.1 Clock & throughput

| Metric | Target |
|--------|--------|
| Clock frequency | ≥ **1.5 GHz** (5 nm) |
| Scalar IPC peak / sustained | 4.0 / 2.5–3.0 |
| **Vector throughput (FP32 elementwise, 1 tile/8 cy)** | **0.77 TFLOPS** |
| **Vector throughput (FP4 elementwise)** | **6.14 TFLOPS** |
| **Vector throughput (FP32 wide row-reduce, recommended baseline)** | **~8.4 GFLOPS effective** (13 beats/8 lanes/iteration) |
| **TINV throughput (128×128 FP32)** | **~1 inverse / 33 µs ≈ 30 K inverses / s @ 1.5 GHz, single-tile-resident** |
| **TMRGSORT throughput (1024 FP32 sort)** | **~6.8 M sorts / s @ 1.5 GHz, single-instruction** |
| Cube FP16 / FP8 / MXFP4 | 12.3 / 24.6 / 98.3 TFLOPS / TOPS |
| MTE tile bandwidth | 4 KB/cy aggregate read + 4 KB/cy aggregate write |
| Memory bandwidth (L2) | 96 GB/s |
| **Mispredict penalty** | **6–7 cy** (vs. v1's 6 cy; the +1 cy is the SSB/STQ tag-CAM propagation) |
| **Speculative depth** | **up to 8 unresolved branches** (matches RAT checkpoint count) |

### 14.2 Workload performance summary

For pure-cube kernels (transformer GEMM, CNN), v2 performance equals v1 (cube unit unchanged). Improvements vs. v1 appear in:

| Workload | v1 | v2 | Speedup |
|----------|-----|----|---------|
| Softmax (batch 8, dim 4096) | 24K cy (vector + scalar) | 18K cy (masked reductions, no `TTRANS` predecessor) | **1.33×** |
| Layer norm (batch 8, dim 4096) | 22K cy | 16K cy | **1.38×** |
| Attention with mask (batch 8, seq 1024) | 80K cy | 56K cy (per-element mask native) | **1.43×** |
| GEMM 128×128 inverse (Kalman update) | software emulation (~4M cy CPU-equivalent on vector) | **TINV 33 K cy** | **~120×** |
| 1024-element top-k | software ~50K cy | TMRGSORT 220 cy + scalar cleanup ~250 cy | **~100×** |
| Speculative scalar-heavy code (e.g. graph algorithm) | limited to in-branch parallelism | full speculation past 8 unresolved branches | **~2–3× sustained IPC improvement** |

### 14.3 IPC breakdown (transformer decode)

```
  Instruction mix (typical transformer layer, M=8, K=4096, N=4096, FP16):
    Scalar:   ~15%
    MTE:      ~25%
    Cube:     ~55%
    Vector:   ~5%
  
  v2 advantage:
    Speculation lets 30–40% of scalar ops past unresolved branches issue early
    → effective scalar IPC rises from ~2.5 (v1) to ~3.2 (v2)
    → end-to-end kernel time drops ~3–5% (cube remains the bottleneck)
```

---

## 15. Area & Power

### 15.1 Area summary (v2 deltas vs. v1 in **bold**)

| Component | v1 area | v2 area | Δ |
|-----------|---------|---------|---|
| TRegFile-4K (1 MB SRAM + 32 b metadata SRAM) | ~1.20 mm² | ~1.20 mm² + ~0.005 mm² metadata | +0.005 mm² |
| outerCube MXU | ~0.80 mm² | ~0.80 mm² | 0 |
| **Vector unit** | ~0.20 mm² (v1) | **~0.30 mm² (VEC-4K-v2 SRAM-staging baseline)** | +0.10 mm² (with new ISA: TINV/TROWRANGE_MUL/TMRGSORT) |
| L1-I / L1-D / L2 | ~0.66 mm² | ~0.66 mm² | 0 |
| Scalar physical RF + Scalar RAT + free list | ~0.22 mm² | ~0.22 mm² | 0 |
| Tile RAT + Tile free list + tile refcount | ~0.05 mm² | ~0.05 mm² + 0.005 mm² metadata RAT | +0.005 mm² |
| Tile Completion Bus (TCB) + tile RS CAMs | ~0.05 mm² | ~0.06 mm² (24-entry vector RS) | +0.01 mm² |
| MTE transpose buffer | 0.005 mm² (4 KB) | 0.001 mm² (512 B) | -0.004 mm² (smaller buffer) |
| RS + dispatch + checkpoint control | ~0.15 mm² | ~0.15 mm² | 0 |
| **Speculative Store Buffer (SSB, 24 entries)** | — | **+0.02 mm²** | +0.02 mm² |
| **Speculative Tile-Store Queue (STQ, 8 entries)** | — | **+0.003 mm²** | +0.003 mm² |
| **Branch-tag tracker** | — | **+0.001 mm²** | +0.001 mm² |
| **Total core (estimated)** | **~3.26 mm²** | **~3.41 mm²** | **+0.15 mm² (+4.6%)** |

### 15.2 Net impact

The v2 core is approximately **5% larger** than v1 in exchange for:

- A re-architected vector unit with per-element masking, 3-source/2-dest, restored FP4/FP8, and three new high-impact instructions (TINV / TROWRANGE_MUL / TMRGSORT).
- Per-port `is_transpose` on TRegFile-4K (eliminating most `TILE.TRANSPOSE` predecessors).
- Branch-prediction-driven speculative execution with full architectural-state recovery (no ROB).

Vector unit area actually shrinks vs. v1's vector unit when accounting for [`vector4k_v2.md`](vector4k_v2.md) §10's analysis (~27% smaller for VEC-v2 SRAM-staging baseline vs. v1) — but the v2 unit also adds the new instructions (TINV / TROWRANGE_MUL / TMRGSORT) that bring net area roughly to parity with v1's vector unit, while delivering ~100× higher performance on those kernels.

### 15.3 Power management

**Same techniques as v1 §14.2.** Adds:

- **Branch-tag tracker clock-gates** when no branches are in flight (very common in straight-line code regions).
- **SSB / STQ entries clock-gate** their flip-flop fields when invalid.

---

## 16. External Interfaces

> **(v1 → v2: 内容未变更,以下完整复制自 v1 §15。)**

### 16.1 Core-to-NoC Interface (v1 §15.1)

| Parameter | Value |
|-----------|-------|
| Bus width | **256 bits** (32 B) |
| Protocol | AXI4 (or similar point-to-point) |
| Outstanding requests | **32** (read) + **16** (write) |
| Burst length | Up to 4 beats (128 B, 2 cache lines) |
| Clock domain | Core clock (synchronous) or async bridge |

### 16.2 Cache Coherence (v1 §15.2)

The Davinci core is designed primarily for single-core or non-coherent multi-core configurations (AI accelerator context). When coherence is needed:

| Parameter | Value |
|-----------|-------|
| Protocol | MOESI or directory-based |
| Snoop filter | L2 tag duplicate |
| Coherence granularity | 64 B (cache line) |

For tile data (TRegFile-4K), coherence is managed at the software level. Tile data bypasses the coherence protocol, flowing through the MTE's dedicated memory path.

### 16.3 Debug & Trace Interface (v1 §15.3)

| Feature | Description |
|---------|-------------|
| Debug halt | External debug request halts core at next instruction boundary |
| PC trace | Compressed branch trace (taken/not-taken stream) |
| Performance counters | 8 programmable counters: IPC, branch mispredict rate, cache miss rate, cube utilization, MTE stalls, RS occupancy |
| Breakpoint registers | 4 instruction address breakpoints + 2 data address watchpoints |

---

## Appendix A: Glossary (v2 additions in **bold**)

| Term | Definition |
|------|-----------|
| RAT | Register Alias Table |
| **Tile Metadata RAT** | 256 × 32 b SRAM holding per-physical-tile (shape.x, shape.y, format) |
| TCB | Tile Completion Bus — 4-port broadcast for tile wakeup |
| CDB | Common Data Bus — broadcast network for scalar results |
| RS | Reservation Station |
| MTE | Memory Tile Engine |
| MXU | Matrix Unit (outerCube) |
| TRegFile-4K | Tile Register File with 4 KB physical tiles, 8R+8W ports, **per-port `is_transpose` (v2)** |
| OPA | Outer Product Accumulate |
| **VEC-4K-v2** | Re-architected vector unit ([`vector4k_v2.md`](vector4k_v2.md)) with staging registers, microcode beat machine, and 3-operand support |
| **SA, SB, SC** | Vector unit's value-tile and mask staging registers (4 KB each, 1R1W SRAM in production baseline) |
| **SX, SY** | Scalar staging slots in VEC-4K-v2 (64 b each, GPR/IMM/TILE/ACC sourced) |
| **TINV** | Tile matrix inverse instruction (up to 128×128 FP32) |
| **TROWRANGE_MUL** | Column-wise product over a dynamic row sub-range |
| **TMRGSORT** | Reconfigurable bitonic sort over any `N = 2^p` up to 8192 |
| **TSETMETA** | Rename-only instruction that updates a tile's metadata word |
| **is_transpose** | Per-read-port flag on TRegFile-4K that selects row-mode vs. col-mode chunk-grid delivery |
| **tilelet_xpose** | Per-beat microcode bit in VEC-4K-v2 selecting per-tilelet chunk-grid transpose at staging-side |
| **branch_tag** | 3-bit tag attached to every µop younger than an unresolved branch; 8 active tags max |
| **SSB** | Speculative Store Buffer — 24-entry buffer that gates scalar stores by branch tag |
| **STQ** | Speculative Tile-Store Queue — 8-entry buffer that gates MTE bulk stores by branch tag |
| **Speculation Tracker** | 5-K-gate structure tracking 8 active branch tags + 8×8 ancestry bitmap |
| ROB | Reorder Buffer — *not present* in Davinci-v2; functionally replaced by RAT checkpoint + refcount + SSB + STQ |
| MSHR | Miss Status Holding Register |
| BTB | Branch Target Buffer |
| TAGE | TAgged GEometric history length predictor |
| RAS | Return Address Stack |
| IPC | Instructions Per Cycle |
| MLP | Memory-Level Parallelism |

## Appendix B: Reference Documents

| Document | Content |
|----------|---------|
| [`Davinci_supersclar.md`](Davinci_supersclar.md) | Davinci v1 — direct predecessor; v2 inherits all unchanged subsystems |
| [`outerCube.md`](outerCube.md) | outerCube MXU architecture, dual-mode operation, ISA, pipeline, performance analysis |
| [`tregfile4k.md`](tregfile4k.md) | TRegFile-4K design (256×4KB tiles, 8R+8W ports, **§7 per-port `is_transpose` enhancement**) |
| [`vector4k_v2.md`](vector4k_v2.md) | **VEC-4K-v2 vector unit specification** — staging registers, per-beat microcode, masked / 3-source / 2-dest, TINV/TROWRANGE_MUL/TMRGSORT |
| [`vector4k.md`](vector4k.md) | VEC-4K v1 vector unit (referenced by v2 for unchanged subsystems) |
| [`Simplified_Superscalar Design Concepts-2.md`](Simplified_Superscalar%20Design%20Concepts-2.md) | OoO execution theory background: no-ROB design, RAT checkpointing, refcount freeing |
| [pto-isa vector docs](https://github.com/hw-native-sys/pto-isa/tree/main/docs/isa) | Authoritative PTO ISA |

## Appendix C: Document History

| Version | Date | Notes |
|---------|------|-------|
| **v2.1** | 2026-04-30 | **Native 3-source ternary FMA family (`VFMA`, `VFNMA`, `VLERP`) added — see §2.2.6a.** Operand `C` is promoted to a **dual role** (mask **or** value tile) selected by a new 1-bit issue-time `c_role ∈ {MASK, VALUE}` flag in the instruction word's `funct6` field (§2.2.2, §2.2.3). With `c_role = VALUE`, `C` is fetched as a full 4 KB value tile through a **3rd VEC-side TRegFile read port (R1)** — TRegFile-4K has 8 read ports, so this is purely a binding allocation, no new SRAM or bank-conflict pressure. With three value tiles fetched in parallel within one 8 cy epoch, `VFMA` runs at the **same throughput as a binary `VADD` / `VMUL` (1 tile / 8 cy)** — a 2× speed-up over the emulated `VMUL` + `VADD` two-instruction sequence — and produces the IEEE-754 single-rounding FMA result, halving the rounding error vs. emulation (critical for FP16 / BF16 / FP8 narrow-format normalisation kernels). **Justification (from [`FMA指令场景说明.md`](FMA指令场景说明.md))**: the canonical `y = γ·x̂ + β` LayerNorm / RMSNorm affine, Welford incremental update (`μ_new = δ·inv_n + μ_old`, `M2_new = δ·δ_2 + M2_old`), Welford state merge, activation polynomials (`gelu`, `swiglu`), and trigonometric polynomials (`sin`, `cos`) all need a third operand that is **not** the previous accumulator — v2.0's `VFMA_ACC D = A·B + Acc` does not apply. **Hardware delta vs. v2.0**: ~6 K gate (~0.2 % of VEC-4K-v2 area) — ~5 K for adding a 512 B/cy value-mode read path on `SC` alongside the existing 1-bit-mask path, ~1 K for control-path widening (Tile RAT / RS / dispatch carry the `c_role` bit). The stage-(B) per-lane FMA core, microcode beat machinery, and 8-port TRegFile already supported `A·B + Z` and the 3rd binding allocation. RS entry width unchanged in concept (the `c_role` bit slots into the existing flags). **Pipeline timing**: §8.3.7 latency table updated with `VFMA / VFNMA` rows (16 cy total = 8 fetch + 8 retire, 1/8 throughput) and `VLERP` (24 cy total, 1/16 throughput due to dual retire); mixed-`is_transpose` rows added (16 cy fetch for one-mismatched, 24 cy for all-distinct degenerate). **Documentation updates**: §2.2.2 operand model gains the `c_role` row + 3rd-port rationale callout; §2.2.3 encoding diagram shows the new `c_role` bit; **new §2.2.6a with full ISA semantics, kernel motivation, hardware-cost breakdown, and pipeline-timing table**; §2.2.8 instruction list gains **Category O — Native 3-source Ternary FMA family**; §8.3.7 latency table updated. **Backward compatibility preserved**: v1 / v2.0 binaries emit `c_role = MASK` exclusively, `R1` stays idle and clock-gated, no behaviour change. Companion update in [`vector4k_v2.md`](vector4k_v2.md) v0.18 (§3.1, §3.3c, §6.2, §7.6, §10). |
| **v2.0** | 2026-04-30 | **Initial Davinci-v2 specification.** Three major changes vs. v1: **(1) TRegFile-4K with per-port `is_transpose` flag** ([`tregfile4k.md`](tregfile4k.md) §7), enabling row-mode or col-mode delivery at full 512 B/cy; consumed by v2 vector unit and (optionally) by cube and MTE. **(2) Vector unit re-architected to VEC-4K-v2** ([`vector4k_v2.md`](vector4k_v2.md)): explicit SRAM-based staging registers (`SA`, `SB`, `SC`) decoupling TRegFile fetch from compute, per-beat microcode dispatch, 3-source / 2-dest tile operands with per-element bitmask predication, restored FP4 and FP8 formats, three new PTO instructions (`TINV` matrix inverse up to 128×128 FP32, `TROWRANGE_MUL` row-range product, `TMRGSORT` bitonic sort over any `N = 2^p` up to 8192), and **tile-register metadata** (32 b: `shape.x`, `shape.y`, `format`) carried via a new Tile Metadata RAT. **(3) Branch-prediction-driven speculative execution** with a ROB-less recovery scheme (§11): a 5-K-gate Branch-Tag Speculation Tracker (8 tags + 8×8 ancestry bitmap), a 24-entry **Speculative Store Buffer** (SSB) that gates scalar stores by branch tag, and an 8-entry **Speculative Tile-Store Queue** (STQ) that gates MTE bulk stores by branch tag. The scheme proves that all three classes of speculative state (renamed registers/tiles, in-flight pipeline state, externally-visible memory effects) can be safely recovered without a Reorder Buffer: classes A and B reuse the v1 RAT-checkpoint + refcount + branch-tag-CAM machinery; class C is gated by SSB / STQ until the producing branch tag becomes non-speculative. Mispredict penalty: 6–7 cy (vs. v1's 6 cy). Total v2 speculation hardware: ~110 K gate (~0.025 mm²), about 3.5% of the v1 core area. v2 core area: ~3.41 mm², a ~5% increase over v1's ~3.26 mm². Performance gains: 1.3–1.4× on masked-vector kernels (softmax, layer norm, masked attention), ~100× on `TINV`-bound (Kalman, NeRF pose) and `TMRGSORT`-bound (top-k, beam-search) kernels, and ~2–3× sustained scalar IPC improvement on speculative-heavy code paths. Cube unit, scalar unit, memory subsystem (caches), and external interfaces remain unchanged from v1. |
