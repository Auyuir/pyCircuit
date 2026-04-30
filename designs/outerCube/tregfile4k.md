### Tile Register File (TRegFile-4K)

The TRegFile-4K is an **8-read / 8-write tile register file** built from **64 physical 1R1W SRAM banks** at **1× core clock**. Storage is organized into **4 KB tiles**, each striped across all 64 banks (64 B per bank) using a **diagonal (skewed) bank map** so that *both* a row-wise sweep and a column-wise sweep of the tile's logical 8×8 chunk grid are bank-conflict-free. Each read port carries an **`is_transpose` bit** that selects between the two delivery orders at run time (§3, §4.2, §7). An **8-cycle synchronized calendar** rotates port-to-group assignments so that every bank sees exactly **1R + 1W per cycle**. Each port accepts one `reg_idx` (plus `is_transpose` on reads) which is latched and drives the next **8-cycle epoch**; a new address is accepted every **8 cycles** (one per epoch boundary), enabling zero-bubble back-to-back tile accesses.

#### 1. Core Parameters

| Parameter | Value |
|-----------|-------|
| SRAM instance | **256 × 512 bits** (64 B wide, depth 256, 1R1W) |
| Banks | **64** (1 SRAM per bank), 8 groups × 8 banks |
| Total size | 64 × 16 KB = **1 MB** |
| Tile size / count | **4 KB** (4096 B) / **256** tiles (tile\_idx 0..255) |
| Read ports | **8** (R0–R7) — 512 B/cy each |
| Write ports | **8** (W0–W7) — 512 B/cy each |
| Calendar | **8 cycles**, synchronized; 1 new `reg_idx` / port / 8 cycles (epoch-aligned) |

#### 2. Tile Layout & Physical Organization

Each 4 KB tile is striped across all 64 banks via a **diagonal (skewed) bank map**. Viewing the 4 KB tile as an **8 × 8 chunk grid** of 64 B chunks, let:

- `g = chunk_offset[5:3]` — **group row** of the chunk grid (0..7).
- `l = chunk_offset[2:0]` — **local col** of the chunk grid within row g (0..7).

The bank select now includes a **3-bit rotator** (not pure wiring); the SRAM address is unchanged:

```
  Skewed bank map (replaces the naive rectangular decode):
    g          = chunk_offset[5:3]         ← group index (chunk-grid row)
    l          = chunk_offset[2:0]         ← local index (chunk-grid col)
    bank_group = g                          ← pure wiring
    bank_local = (l + g) mod 8              ← 3-bit rotator controlled by g
    bank_id    = 8·g + bank_local           ← 0..63
    SRAM_addr  = tile_idx[7:0]              ← unchanged (256 rows / bank)

  Bank groups (8 banks each, unchanged):
    G0 = banks  0– 7    (chunk-grid row 0)
    G1 = banks  8–15    (chunk-grid row 1)
    G2 = banks 16–23    (chunk-grid row 2)
    G3 = banks 24–31    (chunk-grid row 3)
    G4 = banks 32–39    (chunk-grid row 4)
    G5 = banks 40–47    (chunk-grid row 5)
    G6 = banks 48–55    (chunk-grid row 6)
    G7 = banks 56–63    (chunk-grid row 7)

  1 bank  → 64 B   (one chunk)
  1 group → 512 B  (8 banks; one cycle per port in row-mode)
  8 groups → 4 KB  (full tile, 8 cycles)
```

**Diamond placement of a single tile.** Under this skew, the 64 chunks of tile T occupy **one distinct bank in each group row, along a wrapped diagonal** — not a rectangular column-aligned pattern:

```
  (cells list the LOGICAL chunk (g, l) of tile T at each physical bank slot)

                 bank_local = 0      1      2      3      4      5      6      7
    G0 (g=0):              (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)  (0,7)
    G1 (g=1):              (1,7)  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)
    G2 (g=2):              (2,6)  (2,7)  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)
    G3 (g=3):              (3,5)  (3,6)  (3,7)  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)
    G4 (g=4):              (4,4)  (4,5)  (4,6)  (4,7)  (4,0)  (4,1)  (4,2)  (4,3)
    G5 (g=5):              (5,3)  (5,4)  (5,5)  (5,6)  (5,7)  (5,0)  (5,1)  (5,2)
    G6 (g=6):              (6,2)  (6,3)  (6,4)  (6,5)  (6,6)  (6,7)  (6,0)  (6,1)
    G7 (g=7):              (7,1)  (7,2)  (7,3)  (7,4)  (7,5)  (7,6)  (7,7)  (7,0)

  Logical row g = {(g, 0), …, (g, 7)}  — entirely inside group G_g
                                         (local order rotated by g).
  Logical col l = {(0, l), (1, l), …, (7, l)} — one bank per group,
                                                along a wrapped diagonal
                                                (physical locals = (l+0, l+1, …, l+7) mod 8).
```

Every logical row (chunk-grid row) and every logical column (chunk-grid col) therefore touches **exactly one bank in each group** — i.e., both access patterns cover all 64 banks with no bank visited twice. This is the essential property that enables the transposed-read capability in §7.

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  TRegFile-4K:  256 tiles × 4 KB = 1 MB                                               │
 │                                                                                       │
 │  ┌──────────────────────────┐  ┌──────────────────────────┐                           │
 │  │  Group G0 (banks 0–7)    │  │  Group G1 (banks 8–15)   │                           │
 │  │  ┌────┐┌────┐ ... ┌────┐ │  │  ┌────┐┌────┐ ... ┌────┐ │                           │
 │  │  │Bk0 ││Bk1 │     │Bk7 │ │  │  │Bk8 ││Bk9 │     │Bk15│ │                           │
 │  │  │64B ││64B │     │64B │ │  │  │64B ││64B │     │64B │ │                           │
 │  │  │×256││×256│     │×256│ │  │  │×256││×256│     │×256│ │                           │
 │  │  └────┘└────┘     └────┘ │  │  └────┘└────┘     └────┘ │                           │
 │  └──────────────────────────┘  └──────────────────────────┘                           │
 │  ┌──────────────────────────┐  ┌──────────────────────────┐                           │
 │  │  Group G2 (banks 16–23)  │  │  Group G3 (banks 24–31)  │                           │
 │  │  ┌────┐┌────┐ ... ┌────┐ │  │  ┌────┐┌────┐ ... ┌────┐ │                           │
 │  │  │Bk16││Bk17│     │Bk23│ │  │  │Bk24││Bk25│     │Bk31│ │                           │
 │  │  │64B ││64B │     │64B │ │  │  │64B ││64B │     │64B │ │                           │
 │  │  │×256││×256│     │×256│ │  │  │×256││×256│     │×256│ │                           │
 │  │  └────┘└────┘     └────┘ │  │  └────┘└────┘     └────┘ │                           │
 │  └──────────────────────────┘  └──────────────────────────┘                           │
 │  ┌──────────────────────────┐  ┌──────────────────────────┐                           │
 │  │  Group G4 (banks 32–39)  │  │  Group G5 (banks 40–47)  │                           │
 │  │  ┌────┐┌────┐ ... ┌────┐ │  │  ┌────┐┌────┐ ... ┌────┐ │                           │
 │  │  │Bk32││Bk33│     │Bk39│ │  │  │Bk40││Bk41│     │Bk47│ │                           │
 │  │  │64B ││64B │     │64B │ │  │  │64B ││64B │     │64B │ │                           │
 │  │  │×256││×256│     │×256│ │  │  │×256││×256│     │×256│ │                           │
 │  │  └────┘└────┘     └────┘ │  │  └────┘└────┘     └────┘ │                           │
 │  └──────────────────────────┘  └──────────────────────────┘                           │
 │  ┌──────────────────────────┐  ┌──────────────────────────┐                           │
 │  │  Group G6 (banks 48–55)  │  │  Group G7 (banks 56–63)  │                           │
 │  │  ┌────┐┌────┐ ... ┌────┐ │  │  ┌────┐┌────┐ ... ┌────┐ │                           │
 │  │  │Bk48││Bk49│     │Bk55│ │  │  │Bk56││Bk57│     │Bk63│ │                           │
 │  │  │64B ││64B │     │64B │ │  │  │64B ││64B │     │64B │ │                           │
 │  │  │×256││×256│     │×256│ │  │  │×256││×256│     │×256│ │                           │
 │  │  └────┘└────┘     └────┘ │  │  └────┘└────┘     └────┘ │                           │
 │  └──────────────────────────┘  └──────────────────────────┘                           │
 │                                                                                       │
 │  ════════════════════════════════════════════════════════════════════════════════════   │
 │    Rotating group mux: each port gets 1 group per cycle                               │
 │  ════════════════════════════════════════════════════════════════════════════════════   │
 │  ▼(8bk) ▼(8bk) ▼(8bk) ▼(8bk) ▼(8bk) ▼(8bk) ▼(8bk) ▼(8bk)                         │
 │  R0     R1     R2     R3     R4     R5     R6     R7                                  │
 │  512B   512B   512B   512B   512B   512B   512B   512B                                │
 │                                                                                       │
 │  ▲(8bk) ▲(8bk) ▲(8bk) ▲(8bk) ▲(8bk) ▲(8bk) ▲(8bk) ▲(8bk)                         │
 │  W0     W1     W2     W3     W4     W5     W6     W7                                  │
 │  512B   512B   512B   512B   512B   512B   512B   512B                                │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 3. Port Interface

Each port presents **512 B per cycle** (8 banks × 64 B; in row-mode these are the 8 banks of one group, in col-mode these are one bank per group, §4.2). A port accepts one `reg_idx[7:0]` (read ports additionally accept **`is_transpose[0]`**) which is **latched** internally at the epoch boundary. The latched address then drives data delivery (read) or acceptance (write) for the addressed tile over the next **8 consecutive cycles** — one bank-group per cycle per the calendar rotation. Since a 4 KB tile requires 8 × 512 B reads, the port is occupied for the full epoch and can only accept a **new `reg_idx` every 8 cycles**.

- **`is_transpose = 0` (ROW mode)** — default. Cycle-by-cycle the port delivers the chunk-grid rows in order, i.e. a linearly-addressed 512 B strip per cycle. Physical banks visited per cycle = **all 8 banks of one group** (same access pattern as a naive rectangular layout); the 3-bit rotator reorders the 8 lanes so that logical chunk `(g, l)` appears at output lane `l`.
- **`is_transpose = 1` (COL mode)** — delivers the chunk-grid columns in order. Physical banks visited per cycle = **one bank per group** (wrapped diagonal selection). The output lane `i` carries logical chunk `(i, l_active)` where `l_active = (p + cy) mod 8` is the column being scanned this cycle.

Both modes deliver the full 4 KB tile in exactly **8 cycles**; the only difference is the traversal order of the 8 × 8 chunk grid. **Write ports are always row-oriented** — the producer presents 8 chunks of one chunk-grid row per cycle and the write-side rotator places them at the skewed physical banks.

**Epoch-aligned address acceptance:** The port contains a **pending** address register and an **active** address register. A client can write a new `reg_idx` into the pending register at any time during the current epoch. At the next epoch boundary (`cy[2:0]=0`), pending promotes to active and the port begins serving the new tile with **zero bubble**:

```
  Port Rp — back-to-back tile reads (zero gap):

  Cycle:  0    1    2    3    4    5    6    7    8    9   10  ...  15   16  ...
  Addr:  [T0 latched at boundary]              [T1 latched at boundary]  [T2 ...]
  Data:  T0   T0   T0   T0   T0   T0   T0   T0   T1   T1   T1  ...  T1   T2  ...
         .G0  .G1  .G2  .G3  .G4  .G5  .G6  .G7  .G0  .G1  .G2      .G7  .G0
         └──── epoch 0 (tile T0) ────┘  └──── epoch 1 (tile T1) ────┘  └── ...
                                     ↑ zero bubble: T1 starts immediately
```

- T0 address is written to pending before epoch 0; it promotes to active at the boundary.
- T1 address can be written to pending at any point during epoch 0; it takes effect at cycle 8.
- **One new tile address per port every 8 cycles** — the port is fully occupied delivering 512 B/cy × 8 cy = 4 KB for the current tile.

| Ports | Direction | Data Width | Address | Addr Rate |
|-------|-----------|------------|---------|-----------|
| **R0–R7** | Read | 512 B (4096 bits) / cy | `reg_idx[7:0]` + **`is_transpose[0]`** | 1 addr / 8 cy |
| **W0–W7** | Write | 512 B (4096 bits) / cy | `reg_idx[7:0]` + `w_en` | 1 addr / 8 cy |

**Per-port sustained throughput:** 1 tile (4 KB) every 8 cycles = 512 B/cy (row-mode and col-mode both).
**Address registers:** 1 pending + 1 active (double-register for zero-bubble epoch chaining). On read ports `is_transpose` is double-registered together with `reg_idx` and is held constant for the full epoch.

**Port microarchitecture (read port Rp):**

```
             reg_idx[7:0]   is_transpose
                  │               │
                  ▼               ▼
          ┌───────────────────────────┐
          │  Addr+Mode Latch          │◄── written any time during epoch
          │  (pending)                │
          └──────────┬────────────────┘
                     │ epoch boundary: pending → active
                     ▼
          ┌───────────────────────────┐
          │  Addr+Mode Active         │
          │  (current epoch)          │
          └──────────┬────────────────┘
                     │ tile_idx, is_transpose, phase p, cy[2:0]
                     ▼
            ┌──────────────────────────────────────────┐
            │  Bank-Select Calendar (§4)               │
            │    is_transpose = 0 : row-mode           │
            │         → 8 banks of G_{(p+cy) mod 8}    │
            │    is_transpose = 1 : col-mode           │
            │         → bank_i = 8·i + (p+cy+i) mod 8  │
            │           for i ∈ {0..7}                 │
            └──────────┬───────────────────────────────┘
                       │ 8 bank reads × 64 B
                       ▼
            ┌──────────────────────────────────────────┐
            │  Output Rotator / Lane Permute           │
            │    row-mode: rotate-left by g = (p+cy)%8 │
            │              (restore logical l-order)   │
            │    col-mode: identity on group axis      │
            │              (lane i = chunk (i, l_act)) │
            └──────────┬───────────────────────────────┘
                       │ 512 B / cy
                       ▼
                  ┌───────────┐
                  │ data out  │──▶ VEC
                  └───────────┘

  Timing:
    Cycle c     : client writes {reg_idx, is_transpose} → pending latch
    Cycle c'    : next epoch boundary (cy[2:0]=0) → pending promotes to active
    Cycle c'..c'+7 : active tile_idx + mode drive 8 consecutive bank reads
    Cycle c'..c'+7 : client may write next {reg_idx, is_transpose} → new pending
    Cycle c'+8  : next epoch boundary → new pending promotes to active
```

**Write port Wp** is identical except data flows inward and there is no `is_transpose` input. The write-side lane permute is a fixed **rotate-right by g = (p+cy) mod 8** so that the 8 lanes of the chunk-grid row (logical `l = 0..7`) land at physical `bank_local = (l + g) mod 8` in group G_g:

```
                 reg_idx[7:0]              w_en
                      │                     │
                      ▼                     │
              ┌───────────────┐             │
              │  Addr Latch   │◄── client   │
              │  (pending)    │             │
              └──────┬────────┘             │
                     │ epoch boundary       │
                     ▼                      ▼
              ┌───────────────┐     ┌────────────────────────────┐
              │  Addr Active  │────▶│  Bank-Group Sel (cy)       │
              │  (current)    │     │  → group G_{(p+cy) mod 8}  │
              └───────────────┘     └──────────┬─────────────────┘
                                               │ target group g
                                               ▼
                                      ┌────────────────────────┐
           512 B in                   │  Write Lane Rotator    │
         (8 lanes, l=0..7) ──────────▶│  rotate-right by g     │
                                      │  l_phys = (l + g) % 8  │
                                      └──────────┬─────────────┘
                                                 │
                                                 ▼  8 bank writes × 64 B
                                         [SRAM write]
```

#### 4. 8-Cycle Synchronized Calendar

All 16 ports share a global 3-bit **epoch counter** (`cy[2:0]`). Read and write ports follow the **same** base rotation pattern — port *p* (phase offset *p*) is associated with group `(p + cy) % 8` every cycle. Within an epoch a read port then applies one of two bank-select patterns depending on its latched `is_transpose` bit (writes always use the row-mode pattern).

##### 4.1 Row-mode calendar (`is_transpose = 0`, and all writes)

Port *p* at cycle `cy` accesses all 8 banks of group `G_{(p + cy) % 8}`:

| Cycle | Phase 0 (R0/W0) | Phase 1 (R1/W1) | Phase 2 (R2/W2) | Phase 3 (R3/W3) | Phase 4 (R4/W4) | Phase 5 (R5/W5) | Phase 6 (R6/W6) | Phase 7 (R7/W7) |
|-------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 0 | **G0** | **G1** | **G2** | **G3** | **G4** | **G5** | **G6** | **G7** |
| 1 | **G1** | **G2** | **G3** | **G4** | **G5** | **G6** | **G7** | **G0** |
| 2 | **G2** | **G3** | **G4** | **G5** | **G6** | **G7** | **G0** | **G1** |
| 3 | **G3** | **G4** | **G5** | **G6** | **G7** | **G0** | **G1** | **G2** |
| 4 | **G4** | **G5** | **G6** | **G7** | **G0** | **G1** | **G2** | **G3** |
| 5 | **G5** | **G6** | **G7** | **G0** | **G1** | **G2** | **G3** | **G4** |
| 6 | **G6** | **G7** | **G0** | **G1** | **G2** | **G3** | **G4** | **G5** |
| 7 | **G7** | **G0** | **G1** | **G2** | **G3** | **G4** | **G5** | **G6** |

Over 8 cycles each port visits all 8 groups exactly once → reads/writes one complete 4 KB tile in chunk-grid row order.

##### 4.2 Col-mode calendar (`is_transpose = 1`, reads only)

Port *p* at cycle `cy` delivers **chunk-grid column** `l_active = (p + cy) % 8`. The 8 banks accessed are *one per group*, with

```
  for each i ∈ {0..7}:
      bank_i = 8·i + ((p + cy + i) mod 8)       (group G_i, local (l_active + i) mod 8)
```

i.e. every cycle the port fetches a **wrapped-diagonal set of 8 banks**, one from each group. Equivalently, the per-group local-bank table (entries = `bank_local`) for each (phase, cycle) is:

| Cycle | Phase 0 (R0) | Phase 1 (R1) | Phase 2 (R2) | Phase 3 (R3) | Phase 4 (R4) | Phase 5 (R5) | Phase 6 (R6) | Phase 7 (R7) |
|-------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 0 | col 0 → locals (0,1,2,3,4,5,6,7) | col 1 → (1,2,3,4,5,6,7,0) | col 2 → (2,3,4,5,6,7,0,1) | col 3 → (3,4,5,6,7,0,1,2) | col 4 → (4,5,6,7,0,1,2,3) | col 5 → (5,6,7,0,1,2,3,4) | col 6 → (6,7,0,1,2,3,4,5) | col 7 → (7,0,1,2,3,4,5,6) |
| 1 | col 1 | col 2 | col 3 | col 4 | col 5 | col 6 | col 7 | col 0 |
| … | … | … | … | … | … | … | … | … |
| 7 | col 7 | col 0 | col 1 | col 2 | col 3 | col 4 | col 5 | col 6 |

(tuples list `bank_local` for groups `G_0 .. G_7` in order; the cycle-1..7 rows follow the same wrapped-diagonal pattern with `col = (p + cy) mod 8`).

Over 8 cycles each col-mode port visits all 8 columns of the chunk grid exactly once → reads one complete 4 KB tile in chunk-grid column order (i.e. the transpose of the chunk grid).

**Epoch chaining (pipelined address):** The epoch counter is free-running and global. A port's active address drives all 8 cycles of the current epoch. At the next `cy[2:0]=0` boundary, the pending address (latched at any point during the previous epoch) automatically promotes to active. This produces **zero-bubble back-to-back tile accesses** — the port never idles between consecutive tiles:

```
  cy[2:0]: 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7  0  1 ...
  Active:  ──── tile T0 ─────────  ──── tile T1 ─────────  ── T2 ...
  Pending:       [T1 latched]            [T2 latched]
                                  ↑                        ↑
                           T1 promotes                T2 promotes
```

**Conflict-free proof (row-mode, `is_transpose = 0` for all reads):** At every cycle, the 8 read ports cover {G0..G7} and the 8 write ports independently cover {G0..G7}. Each group sees exactly 1R + 1W. The reader and writer assigned to the same group are always the **same-phase** pair (R0/W0, R1/W1, ..., R7/W7).

```
  Cy 0: R = G0(R0) G1(R1) G2(R2) G3(R3) G4(R4) G5(R5) G6(R6) G7(R7)
         W = G0(W0) G1(W1) G2(W2) G3(W3) G4(W4) G5(W5) G6(W6) G7(W7)
  Cy 1: R = G1(R0) G2(R1) G3(R2) G4(R3) G5(R4) G6(R5) G7(R6) G0(R7)
         W = G1(W0) G2(W1) G3(W2) G4(W3) G5(W4) G6(W5) G7(W6) G0(W7)
  ...
  Cy 7: R = G7(R0) G0(R1) G1(R2) G2(R3) G3(R4) G4(R5) G5(R6) G6(R7)
         W = G7(W0) G0(W1) G1(W2) G2(W3) G3(W4) G4(W5) G5(W6) G6(W7)

  Per bank: ≤ 1R + 1W per cycle.  Two-port SRAM satisfied.  ✓
```

**Conflict-free proof (col-mode, `is_transpose = 1` for all reads):** At cycle `cy`, read port R_p accesses bank `(G_i, (p + cy + i) mod 8)` for each `i ∈ {0..7}`. At a given group `G_i` and cycle `cy`, as `p` ranges over `{0..7}`, the local index `(p + cy + i) mod 8` is a **bijection** onto `{0..7}`. Hence the 8 read ports together cover every bank of every group exactly once — 64 reads, all distinct. Writes (always row-mode) still cover {G0..G7} once each; each bank sees ≤ 1R + ≤ 1W per cycle.

```
  Cy 0 col-mode reads (bank_id of each port = 8·i + local):
      R0 → 0, 9,18,27,36,45,54,63       (col 0: locals 0,1,…,7)
      R1 → 1,10,19,28,37,46,55,56       (col 1: locals 1,2,…,7,0)
      R2 → 2,11,20,29,38,47,48,57       (col 2: locals 2,3,…,7,0,1)
      R3 → 3,12,21,30,39,40,49,58       (col 3)
      R4 → 4,13,22,31,32,41,50,59       (col 4)
      R5 → 5,14,23,24,33,42,51,60       (col 5)
      R6 → 6,15,16,25,34,43,52,61       (col 6)
      R7 → 7, 8,17,26,35,44,53,62       (col 7)

  Each physical bank 0..63 appears in exactly one port list ⇒ no conflict. ✓
  Writes (all row-mode) land on the same 8 groups with same-phase bypass. ✓
```

**Mixed-mode across read ports is forbidden** — see §6 scheduling rule and §7 proof of the collision.

#### 5. Throughput

| Metric | Value |
|--------|-------|
| Per port data BW | 8 banks × 64 B = **512 B/cy** (row-mode and col-mode) |
| Per port per epoch (8 cy) | 8 chunk-grid rows **or** 8 chunk-grid cols × 512 B = **4 KB** (1 tile) |
| Addr acceptance rate | **1 `reg_idx` (+ `is_transpose` on reads) / port / 8 cycles** (epoch-aligned) |
| Addr-to-data latency | 0–7 cy (depends on when within epoch the pending addr/mode is written) |
| Sustained tile rate | 1 tile / 8 cy / port (zero-bubble epoch chaining) |
| Aggregate read BW | 8 ports × 512 B/cy = **4 KB/cy** (either all row-mode or all col-mode, §6) |
| Aggregate write BW | 8 ports × 512 B/cy = **4 KB/cy** |
| Total per epoch | **16 tile ops** (8R + 8W), zero bank conflicts |
| Transpose cost | **0 cycles** — a col-mode read delivers the chunk-grid transpose at full 512 B/cy without any extra latency, storage, or copy |

#### 6. Write-to-Read Bypass & Scheduling Constraint

**Same-phase bypass (hardware, zero-latency):**

The calendar guarantees that each group's reader and writer in any given cycle are always a same-phase port pair. When a same-phase read and write target the same `tile_idx`, SRAM write latency (1 cycle) would return stale data. A combinational bypass mux forwards the write data directly to the read output.

**Cross-phase RAW hazard (not resolved in hardware):**

For different-phase port pairs (e.g. R0 reading tile T while W1 writes tile T in the same epoch), the phase offset causes one or more groups per pair to be **read before they are written**. The write data does not exist at the time of the read, so no combinational bypass can resolve it.

Example — R0 (phase 0) and W1 (phase 1) on the same tile:

```
  Group  │  R0 reads  │  W1 writes  │  Result
  ───────┼────────────┼─────────────┼──────────────────────────
  G0     │  cycle 0   │  cycle 7    │  Read 7 cy before write → STALE ✗
  G1     │  cycle 1   │  cycle 0    │  Write 1 cy before read → SRAM OK ✓
  G2     │  cycle 2   │  cycle 1    │  Write 1 cy before read → SRAM OK ✓
  G3     │  cycle 3   │  cycle 2    │  Write 1 cy before read → SRAM OK ✓
  G4     │  cycle 4   │  cycle 3    │  Write 1 cy before read → SRAM OK ✓
  G5     │  cycle 5   │  cycle 4    │  Write 1 cy before read → SRAM OK ✓
  G6     │  cycle 6   │  cycle 5    │  Write 1 cy before read → SRAM OK ✓
  G7     │  cycle 7   │  cycle 6    │  Write 1 cy before read → SRAM OK ✓
```

**Scheduling rules (enforced by upstream scheduler):**

> **(R1)** Within the same 8-cycle epoch, no two different-phase read/write ports shall operate on the same `tile_idx`. Same-phase pairs (R0/W0, R1/W1, ..., R7/W7) are always safe and fully bypassed. Cross-phase pairs on the same tile must be separated by at least one full epoch (8 cycles).
>
> **(R2) Uniform transpose mode per epoch.** Within the same 8-cycle epoch, **all 8 active read ports must share the same `is_transpose` value**. Row-mode and col-mode reads cannot coexist in the same epoch because a row-mode reader occupies all 8 banks of its group G_a while every col-mode reader simultaneously wants exactly one bank inside G_a — the two patterns collide on the 1R SRAM port of that bank.
>
> Because `is_transpose` is double-registered on a per-port basis, each port *can* switch between row and col across successive epochs, but the scheduler must ensure the 8 active reads of any given epoch agree. Writes are always row-mode and impose no new constraint.

**Why mixed-mode is disallowed (sketch).** At cycle `cy`, a row-mode port R_p occupies *all* 8 banks of group `G_{(p+cy) mod 8}`. A col-mode port R_q at the same cycle needs the bank at `(G_i, local = (q + cy + i) mod 8)` for every `i`, including `i = (p + cy) mod 8`, which collides with the group R_p has fully claimed. Since each SRAM bank has only 1R port, the collision is unresolvable by rotation or reorder. The uniform-mode rule sidesteps this cleanly. For row-mode + row-mode or col-mode + col-mode, the bijection arguments in the §4 proofs guarantee zero overlap.

#### 7. Transposed Read — Diagonal Skew, Datapath, and Semantics

This section consolidates the **transposed-read enhancement**: how the bank-skew of §2 together with the `is_transpose` bit on the read port (§3) turns the TRegFile into a bank-conflict-free *row-or-column* tile fetcher, at a small fixed datapath cost and without any extra SRAM storage, redundancy, or latency.

##### 7.1 Motivation

Many tile operations — matrix transpose, GEMM lhs/rhs reshape, strided reductions across "columns", butterfly shuffles, etc. — need to consume the **columns** of an R × C tile at the same bandwidth they consume its rows. A naive rectangular bank decode (`bank = chunk_offset[5:0]`) is ideal for row-major sweeps but forces all 8 chunks of a logical column into the *same bank group*, causing a 1× → 1/8× bandwidth collapse plus a hard bank conflict.

The diagonal skew solves this with one small hardware change (a per-group 3-bit rotator on writes, a matching inverse rotator plus a bank-select mux on reads) and **zero cost** in throughput, latency, storage, or port count.

##### 7.2 Recap of the skewed bank map

From §2:

```
  For chunk (g, l) of tile T, with g = chunk_offset[5:3], l = chunk_offset[2:0]:
      bank_id    = 8·g + ((l + g) mod 8)
      SRAM_addr  = tile_idx[7:0]
```

The mapping is:

- **Injective within a tile** — the 64 chunks of one tile occupy 64 distinct banks (one per physical bank).
- **Row-closed** — all 8 chunks of chunk-grid row `g` lie in the same group `G_g`, just rotated in local order.
- **Column-spread** — the 8 chunks of chunk-grid column `l` lie one-per-group, along a wrapped diagonal (`bank_local = (l + g) mod 8` for `g = 0..7`).

Both properties together give the **row × col conflict-free** guarantee.

##### 7.3 Write datapath (always row-oriented)

The producer presents each 512 B strip as 8 logical lanes (`l = 0..7`) of chunk-grid row `g` (chosen by the calendar from §4.1). A fixed **3-bit write-lane rotator** steered by `g` places logical lane `l` at physical `bank_local = (l + g) mod 8`:

```
    logical lanes  l=0 1 2 3 4 5 6 7   (512 B in)
                       │ │ │ │ │ │ │ │
                       ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
                  ╔════════════════════╗
                  ║  Rotate-right by g ║  g = (p + cy) mod 8  (from calendar)
                  ╚════════════════════╝
                       │ │ │ │ │ │ │ │
                       ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
    physical       bank_local = 0 1 2 3 4 5 6 7  of group G_g
    banks          (SRAM_addr = tile_idx)
```

The rotator is 8-way, 64 B wide, with a 3-bit select — one instance per write port (8 total). No extra storage, no per-bank decode.

##### 7.4 Read datapath (row-mode *or* col-mode)

Read ports implement both traversal orders over a **shared bank-select mux** steered by `{phase p, cy[2:0], is_transpose}`. A 9-to-8 lane permute on the output side restores logical order.

```
                 ┌──────── row-mode ────────┐    ┌──────── col-mode ────────┐
                 │  fetch 8 banks of G_g    │    │  fetch 1 bank per group  │
    Bank sel:    │   g = (p + cy) mod 8     │    │   bank_i = 8·i           │
                 │   bank_i = 8·g + i       │    │          + (p+cy+i) mod 8│
                 │   (i = local 0..7)       │    │   (i = group 0..7)       │
                 └──────────────────────────┘    └──────────────────────────┘
                                │                            │
                                └─────────┬──────────────────┘
                                          ▼
                           ┌──────────────────────────────┐
                           │   Output Lane Permute        │
                           │   row-mode: rotate-left  g   │
                           │   col-mode: identity         │
                           └──────────────┬───────────────┘
                                          ▼
                                   512 B to VEC
```

- **Row-mode output meaning.** Output lane `l` carries logical chunk `(g_active, l)`; strips arrive in chunk-grid-row order over the 8 cycles of the epoch (bytes `0..4095` of the tile in linear address order, modulo the phase offset).
- **Col-mode output meaning.** Output lane `i` carries logical chunk `(i, l_active)` with `l_active = (p + cy) mod 8`; strips arrive in chunk-grid-*column* order — i.e., **the transpose of the 8 × 8 chunk grid**, still delivered at full 512 B/cy.

The row-mode output rotator is the inverse of the write-side rotator (rotate-left by `g` vs. rotate-right by `g`); col-mode needs no lane permute because the fetched 8-tuple is already `chunk (0, l), chunk (1, l), …, chunk (7, l)` in group order.

##### 7.5 Semantics: what "transpose" means at the TRegFile level

The skew provides a **chunk-granular transpose** of the tile, not a byte- or element-granular one. Specifically:

> Col-mode reads deliver the 4 KB tile as the **transpose of its 8 × 8 chunk-grid partition**. The 64 B inside each chunk are *not* transposed — they come out of the SRAM in their stored byte order.

The TRegFile deliberately does **not** attempt an element-level transpose; finer permutation belongs to the VEC front-end, which already has a 512 B Align/Unpack/Permute stage.

**Strip-fill invariant.** Both row-mode and col-mode always deliver the full **512 B** per cycle. Col-mode has a clean physical description (derived directly from the bank map):

> **Col-mode strip `s` = the eight 64 B windows at tile byte positions `{ s·64 + k·512 : k = 0..7 }`.**

Equivalently, col-mode picks *the same 64 B sub-chunk (offset `s·64`) from each of the 8 row-mode strips*, and delivers them concatenated as one 512 B strip.

**Full-tile coverage invariant (independent of shape).** Over an 8-cycle epoch `s = 0..7`, col-mode covers chunk-id sets
`{ 8·g + s : g = 0..7 }` which are disjoint for distinct `s` and whose union is exactly `{0, 1, …, 63}`. The map `(s, g) ↦ 8·g + s` is a **bijection** onto the 64 chunks of the tile; therefore *every* col-mode read delivers the **complete 4096 B of the tile in 8 cycles — no duplicate, no omission — for every one of the 23 legal `(E, R, C)` shapes**. The ✅ markings below only classify whether the delivery is semantically a clean whole-row transpose view (✅) or an interleaved row-segment view that VEC's Align/Unpack/Permute front-end must recombine (no ✅); the *data completeness* is identical in both cases.

With the coverage and strip-fill formulas, for any legal `(R, C, E)` with row width `W = C·E` bytes, the set of tile rows touched by col-mode strip `s` is

```
  touched-rows(s) = { ( s·64 + k·512 ) div W :  k = 0..7 }     (as a multiset if W < 64)
  byte-offset-in-row(s, k) = ( s·64 + k·512 ) mod W
```

and the 512 B payload is exactly `512 / W` rows' worth of data (whole rows when `W ≤ 64`, row-segments otherwise). The following regime table and the shape-by-shape enumeration are direct corollaries.

| `W` range | Row-mode strip contents | Col-mode strip contents |
|-----------|-------------------------|-------------------------|
| `W ≤ 64` (row fits in 1 chunk, `64/W` rows per chunk) | `512/W` contiguous rows | `512/W` **whole** rows, arranged as `8` groups of `64/W` consecutive rows, with group `k` centered at base row `k·(R/8) + s·(64/W)` |
| `W = 128, 256, 512` (row = 2, 4, 8 chunks) | `512/W` contiguous rows | 8 × 64 B *row-segments* picked from 8 distinct rows, stride `R/8`; each segment covers one of the `W/64` sub-chunks of its row, cycled by `s` |
| `W = 1024, 2048, 4096` (one row spans `W/512` row-mode strips) | `1 / (W/512)` row (a half / quarter / eighth) | 8 × 64 B row-segments drawn from the `R ≤ 4` rows of the tile, at stride-512 byte positions inside each row |

**Per-shape coverage across all 23 legal `(E, R, C)` geometries** of [`vector4k.md`](vector4k.md) §9.7 (FP16 and BF16 share shapes):

| `R × C` (FP32, `E=4`) | `W=C·E` | Row-mode rows / strip | Col-mode strip `s` delivers | `R × C` (FP16 / BF16, `E=2`) | `W` | Row-mode rows / strip | Col-mode strip `s` delivers |
|-----------------------|---------|-----------------------|-----------------------------|------------------------------|-----|-----------------------|-----------------------------|
| 1 × 1024 | 4096 B | `1/8` row | 8 × 64 B of the sole row, stride 512 B (= 128 FP32 stride-128) | 1 × 2048 | 4096 B | `1/8` row | 8 × 64 B of the sole row, stride 512 B (= 256 FP16 stride-256) |
| 2 × 512  | 2048 B | `1/4` row | 4 × 64 B of row 0 + 4 × 64 B of row 1, stride 512 B inside each row | 2 × 1024 | 2048 B | `1/4` row | same, scaled by `E` |
| 4 × 256  | 1024 B | `1/2` row | 2 × 64 B of each of 4 rows, stride 512 B inside each row | 4 × 512  | 1024 B | `1/2` row | same, scaled by `E` |
| **8 × 128**  | **512 B** | **1 whole row** | **1 × 64 B of each of 8 rows** ⇒ **clean 8 × 8 chunk-grid transpose = 8 rows × 16 FP32 col-band** ✅ | **8 × 256**  | **512 B** | **1 whole row** | **8 rows × 32 FP16 col-band** ✅ |
| 16 × 64  | 256 B  | 2 rows | 8 rows (stride 2: rows {0,2,…,14} for s ∈ {0..3}; rows {1,3,…,15} for s ∈ {4..7}), each at one of 4 col-bands of 16 FP32 | 16 × 128 | 256 B  | 2 rows | same pattern over 16 rows × 128 FP16 |
| 32 × 32  | 128 B  | 4 rows | 8 rows (stride 4), each at one of 2 half-row col-bands of 16 FP32 | 32 × 64  | 128 B  | 4 rows | same over 32 rows × 64 FP16 |
| **64 × 16** | **64 B**  | **8 whole rows** | **8 complete rows (stride 8 over 64 rows)** ✅ | **64 × 32** | **64 B**  | **8 whole rows** | **8 complete rows (stride 8 over 64 rows)** ✅ |
| 128 × 8  | 32 B   | 16 whole rows | 8 × (2 consecutive rows) with stride 16 over 128 rows | 128 × 16 | 32 B   | 16 whole rows | 8 × (2 consecutive rows), stride 16 over 128 rows |
| 256 × 4  | 16 B   | 32 whole rows | 8 × (4 consecutive rows) with stride 32 over 256 rows | 256 × 8  | 16 B   | 32 whole rows | 8 × (4 consecutive rows), stride 32 over 256 rows |
| 512 × 2  | 8 B    | 64 whole rows | 8 × (8 consecutive rows) with stride 64 over 512 rows | 512 × 4  | 8 B    | 64 whole rows | 8 × (8 consecutive rows), stride 64 over 512 rows |
| 1024 × 1 | 4 B    | 128 whole rows | 8 × (16 consecutive rows) with stride 128 over 1024 rows | 1024 × 2 | 4 B    | 128 whole rows | 8 × (16 consecutive rows), stride 128 over 1024 rows |
| — | | | | 2048 × 1 | 2 B    | 256 whole rows | 8 × (32 consecutive rows), stride 256 over 2048 rows |

(✅ marks the two regimes in which col-mode produces a clean *whole-row* delivery: `W = 512 B` → eight full rows laid out as a 8 × 8 col-band transpose, and `W ≤ 64 B` → `512/W` full rows chosen as eight stride-`R/8` row-blocks. In every other regime col-mode still fills the 512 B port, but with row-segments instead of whole rows — VEC's Align/Unpack/Permute recombines them.)

**Universal bank-conflict-freeness (proof).** The per-cycle physical bank set accessed by the 8 read ports is

```
  row-mode (∀ p ∈ {0..7}):  { 8·g + ((l+g) mod 8) : l = 0..7 }  with g = (p+cy) mod 8
  col-mode (∀ p ∈ {0..7}):  { 8·i + ((p+cy+i) mod 8) : i = 0..7 }
```

Neither expression depends on `R`, `C`, `E`, `tile_idx`, or any tile content; both are functions only of `(p, cy)`. The two §4 bijection arguments therefore hold for **every** `(E, R, C)` in the 23-row legal-shape table unchanged. Writes (always row-mode) are likewise shape-agnostic. The **"uniform transpose mode per epoch"** rule (§6 R2) is the single necessary and sufficient scheduling constraint; no shape introduces any extra hazard.

Hence:

> **Transposed read is bank-conflict-free for all 23 legal `(E, R, C)` geometries and at every valid `W = C·E` from 2 B to 4096 B.** The 512 B read port is fully filled on every cycle; for `W ≤ 64` it is filled with `512/W` whole rows (stride-`R/8` gather), for `W ∈ (64, 512]` with `512/W` rows' worth of 64 B row-segments, and for `W > 512` with the same fractional-row budget as row-mode.

VEC's existing Align/Unpack/Permute front-end (`vector4k.md` §4) is the sole agent that lifts these strip-level deliveries to the element-level transpose required by `TCOL*` / `TTRANS` class instructions — the TRegFile guarantees only that the raw 4 KB of data arrives, bank-conflict-free, in the chunk-grid-transposed order.

##### 7.6 Datapath cost

| Block | Before | After | Δ |
|-------|--------|-------|---|
| Bank decode | pure wiring | 3-bit rotator controlled by `g` | 8 × 64 B 8-way rotators per port (8 read + 8 write) |
| Bank-select mux | 1 option (row-mode) | 2 options (row or col) steered by `is_transpose` | small 2-to-1 mux per port, plus col-mode address generator (`bank_i = 8·i + (p+cy+i) mod 8`) |
| Output lane permute (read) | none | rotate-left by `g` (row-mode) / identity (col-mode) | 8 × 64 B 8-way rotators per read port |
| Storage | — | **no change** | 0 extra SRAM, 0 extra rows |
| Ports | 8R + 8W | **no change** | 0 extra ports |
| Latency | read latency of underlying SRAM | **no change** | rotator + mux are combinational and fit inside the existing pipeline stage |
| `is_transpose` registers | — | 1 bit of pending + 1 bit of active per read port | 16 FFs total |

Net area overhead is dominated by the 24 × 8-way 64 B-wide rotators (8 write + 8 read input-side + 8 read output-side), which at a 64 B (512-bit) granularity are standard building blocks at negligible cost compared to the 64 × (256 × 512 b) SRAM macros.

##### 7.7 Worked example — chunks of one tile

Assume `tile_idx = 0x2A`, producer writes the 4 KB tile with the natural row-major linear address ordering (chunk 0 at bytes 0..63, chunk 63 at bytes 4032..4095). After the write-side rotator, physical SRAM contents at SRAM row `0x2A` are:

```
  bank  0  (G0,l=0)  ← chunk (0,0)      bank  8  (G1,l=0)  ← chunk (1,7)
  bank  1  (G0,l=1)  ← chunk (0,1)      bank  9  (G1,l=1)  ← chunk (1,0)
  bank  2  (G0,l=2)  ← chunk (0,2)      bank 10  (G1,l=2)  ← chunk (1,1)
  …                                     …
  bank  7  (G0,l=7)  ← chunk (0,7)      bank 15  (G1,l=7)  ← chunk (1,6)

  bank 16  (G2,l=0)  ← chunk (2,6)      bank 24  (G3,l=0)  ← chunk (3,5)
  bank 17  (G2,l=1)  ← chunk (2,7)      bank 25  (G3,l=1)  ← chunk (3,6)
  …                                     …
```

A subsequent **row-mode** read on port R0 (phase 0) delivers, cycle-by-cycle:

```
  cy=0: chunk (0,0) (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (0,7)   ← group G0, rotator = 0
  cy=1: chunk (1,0) (1,1) (1,2) (1,3) (1,4) (1,5) (1,6) (1,7)   ← group G1, rotator = 1
  …
  cy=7: chunk (7,0) (7,1) (7,2) (7,3) (7,4) (7,5) (7,6) (7,7)   ← group G7, rotator = 7
```

i.e. the tile in natural row-major order (bytes 0..4095).

A **col-mode** read on port R0 delivers:

```
  cy=0: chunk (0,0) (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0)   ← col 0
  cy=1: chunk (0,1) (1,1) (2,1) (3,1) (4,1) (5,1) (6,1) (7,1)   ← col 1
  …
  cy=7: chunk (0,7) (1,7) (2,7) (3,7) (4,7) (5,7) (6,7) (7,7)   ← col 7
```

i.e. the **transpose of the 8 × 8 chunk grid**, still 4 KB in 8 cycles, still conflict-free, with the only per-cycle physical access being a wrapped-diagonal bank set as listed in the §4.2 proof.

##### 7.8 Summary of the enhancement

1. **Storage layout** is a diagonal/diamond skew (`bank = 8·g + (l + g) mod 8`); the SRAM address and bank count are unchanged.
2. **Read ports** gain an `is_transpose` input (double-registered with `reg_idx`); a port can pick row or col delivery order at each epoch boundary.
3. **Calendar** is unchanged in its port-to-phase rotation; per-cycle bank-select is either *whole group* (row-mode) or *one bank per group* (col-mode).
4. **Write ports** are unchanged externally; a fixed 3-bit rotator places written lanes at the skewed physical bank positions.
5. **Throughput, latency, storage, SRAM port count**: all **unchanged**.
6. **Scheduling** gains one rule: all 8 active reads of any epoch share the same `is_transpose`. This pairs naturally with the existing "1 address per port per 8 cycles" cadence.
7. **Gain**: true bank-conflict-free transpose of the 8 × 8 chunk partition of any tile, delivered at full 512 B/cy — no spare cycles, no duplicate storage, no side buffer.

#### 8. Revision History

| Version | Notes |
|---------|-------|
| 0.1 | Initial design: rectangular bank decode, row-mode only |
| **0.2** | **Diagonal skew + `is_transpose` read input: bank-conflict-free row-*or*-col tile delivery at full 512 B/cy. Added §7 detailing storage map, read/write datapath, scheduling rule (R2), cost, and worked examples.** |

