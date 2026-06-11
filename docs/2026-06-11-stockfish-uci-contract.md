Parent: none

# Stockfish UCI I/O — exact format spec

Source: `src/llm/runtime/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe` → **Stockfish 18**. All lines below captured live 2026-06-11. Match these byte formats exactly.

Transport: line-based ASCII over stdin/stdout, each line `\n`-terminated, flush per line. `go` is async (engine keeps reading stdin during search).

Notation: `<x>` = required field, `[x]` = optional, `…` = repeatable, `|` = alternative.

---

## On launch (before any input)

```
Stockfish 18 by the Stockfish developers (see AUTHORS file)
```

---

## INPUT (GUI → engine)

```
uci
isready
ucinewgame
setoption name <id> [value <v>]
position startpos [moves <m> …]
position fen <FEN> [moves <m> …]
go <limit> …
stop
ponderhit
quit
d
eval
```

`go` limits (any combination, space-separated):
```
depth <n>
movetime <ms>
nodes <n>
wtime <ms> btime <ms> winc <ms> binc <ms> [movestogo <n>]
infinite
mate <n>
ponder
searchmoves <m> …
perft <n>
```

Move format (`<m>`): long algebraic `<from><to>[promo]`. Promo = lowercase `q|r|b|n`. Castling = king move (`e1g1`, `e1c1`). Null = `0000`.

---

## OUTPUT (engine → GUI)

### Response to `uci`
```
id name Stockfish 18
id author the Stockfish developers (see AUTHORS file)

option name <id> type <spin|check|string|button|combo> default <v> [min <n> max <n>]
…
uciok
```
Full option list emitted by SF18 (advertise only what you honor):
```
option name Debug Log File type string default <empty>
option name NumaPolicy type string default auto
option name Threads type spin default 1 min 1 max 1024
option name Hash type spin default 16 min 1 max 33554432
option name Clear Hash type button
option name Ponder type check default false
option name MultiPV type spin default 1 min 1 max 256
option name Skill Level type spin default 20 min 0 max 20
option name Move Overhead type spin default 10 min 0 max 5000
option name nodestime type spin default 0 min 0 max 10000
option name UCI_Chess960 type check default false
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 1320 min 1320 max 3190
option name UCI_ShowWDL type check default false
option name SyzygyPath type string default <empty>
option name SyzygyProbeDepth type spin default 1 min 1 max 100
option name Syzygy50MoveRule type check default true
option name SyzygyProbeLimit type spin default 7 min 0 max 7
option name EvalFile type string default nn-c288c895ea92.nnue
option name EvalFileSmall type string default nn-37f18f62d772.nnue
```

### Response to `isready`
```
readyok
```

### Response to `ucinewgame`, `setoption`, `position`
None. (Some `setoption` may emit an `info string …`.)

### Response to `go` (search): zero+ `info` lines, then exactly one `bestmove`

`info` line — fields in this exact order, omit any not computed:
```
info depth <n> seldepth <n> multipv <n> score <cp <n> | mate <n>> [wdl <w> <d> <l>] [lowerbound|upperbound] nodes <n> nps <n> hashfull <n> tbhits <n> time <ms> pv <m> …
```
Real examples:
```
info depth 12 seldepth 19 multipv 1 score cp 47 nodes 41065 nps 821300 hashfull 11 tbhits 0 time 50 pv e2e4 c7c5 b1c3 a7a6 g1f3
info depth 1 seldepth 2 multipv 1 score mate 1 nodes 20 nps 20000 hashfull 0 tbhits 0 time 1 pv a1a8
info depth 10 seldepth 12 multipv 1 score cp 69 wdl 235 763 2 nodes 4835 nps 805833 hashfull 1 tbhits 0 time 6 pv e2e4 e7e5 g1f3 b8c6
```
- `score cp <n>`: centipawns, side-to-move POV (positive = mover better).
- `score mate <n>`: mate in `n` moves; negative = being mated.
- `wdl <w> <d> <l>`: only if `UCI_ShowWDL true`; per-mille, sums 1000; placed immediately after score.
- MultiPV >1: one line per ranked move, `multipv 1`, `multipv 2`, … (each `go` iteration).
- `info string <text>`: free-form diagnostics, GUI ignores.

`bestmove` — terminates the search:
```
bestmove <m> [ponder <m>]
```
No legal move:
```
bestmove (none)
```

### Response to `go perft <n>`: per-move counts, blank line, total. No `bestmove`.
```
<m>: <count>
…

Nodes searched: <total>
```
Example (`perft 2` from startpos):
```
a2a3: 20
b2b3: 20
…
g1h3: 20

Nodes searched: 400
```

### Response to `stop`
Same as a finished `go`: final `bestmove <m> [ponder <m>]`.

### Response to unknown command
```
Unknown command: '<input>'. Type help for more information.
```

### Response to `d` (debug, non-UCI)
```
 +---+---+---+---+---+---+---+---+
 | r | n | b | q | k | b | n | r | 8
 +---+---+---+---+---+---+---+---+
 …
   a   b   c   d   e   f   g   h

Fen: <FEN>
Key: <hex>
Checkers: <squares>
```

### Response to `eval` (debug, non-UCI)
```
NNUE evaluation        +0.12 (white side)
Final evaluation       +0.15 (white side) [with scaled NNUE, ...]
```

---

## Mandatory subset (must implement)

| Input | Output |
|---|---|
| `uci` | `id name`, `id author`, `option …`*, `uciok` |
| `isready` | `readyok` |
| `ucinewgame` | none |
| `setoption …` | none |
| `position …` | none |
| `go …` | `info …`*, `bestmove <m> [ponder <m>]` |
| `stop` | `bestmove <m> [ponder <m>]` |
| `quit` | exit |

Optional: `ponder`, `ponderhit`, `perft`, `d`, `eval`, `wdl`, MultiPV.

---

## Not reproducible on this hardware (standard UCI, include if implementing)

- `currmove <m> currmovenumber <n>` — SF emits only when a root move's search exceeds ~3s; binary runs >1M nps so it never fired.
- `lowerbound` / `upperbound` — aspiration-window fail-high/low flag on the `info` line; short searches didn't trigger.
