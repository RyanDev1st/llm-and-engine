Parent: none

# Stockfish UCI I/O — exact format spec

Source: `src/llm/runtime/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe` → **Stockfish 18**. All formats captured live 2026-06-11. Match exactly.

Transport: line-based ASCII over stdin/stdout, each line `\n`-terminated, flush per line. `go` is async (engine keeps reading stdin during search).

Notation: `<x>` required, `[x]` optional, `…` repeatable, `a|b` alternative.

---

## On launch (before any input)

| Line emitted |
|---|
| `Stockfish 18 by the Stockfish developers (see AUTHORS file)` |

---

## INPUT commands (GUI → engine)

| Command | Format | Engine response |
|---|---|---|
| uci | `uci` | id lines + option lines + `uciok` |
| isready | `isready` | `readyok` (reply anytime, even mid-search) |
| ucinewgame | `ucinewgame` | none |
| setoption | `setoption name <id> [value <v>]` | none (may emit `info string …`) |
| position (start) | `position startpos [moves <m> …]` | none |
| position (fen) | `position fen <FEN> [moves <m> …]` | none |
| go | `go <limit> …` | `info` lines + one `bestmove` |
| stop | `stop` | final `bestmove` |
| ponderhit | `ponderhit` | continue search → `bestmove` |
| quit | `quit` | exit process |
| d | `d` | board dump (debug, non-UCI) |
| eval | `eval` | eval dump (debug, non-UCI) |
| *unknown* | any other | `Unknown command: '<input>'. Type help for more information.` |

### `go` limit tokens (combine any, space-separated)

| Token | Format | Meaning |
|---|---|---|
| depth | `depth <n>` | search to fixed depth |
| movetime | `movetime <ms>` | search fixed milliseconds |
| nodes | `nodes <n>` | stop after n nodes |
| clock | `wtime <ms> btime <ms> winc <ms> binc <ms> [movestogo <n>]` | manage own time |
| infinite | `infinite` | search until `stop` |
| mate | `mate <n>` | search for mate in n |
| ponder | `ponder` | ponder mode (ends on `ponderhit`/`stop`) |
| searchmoves | `searchmoves <m> …` | restrict to listed root moves |
| perft | `perft <n>` | move-count test (no `bestmove`) |

### Move format `<m>`

| Case | Format | Example |
|---|---|---|
| normal | `<from><to>` | `e2e4` |
| promotion | `<from><to><piece>` (piece = `q\|r\|b\|n`, lowercase) | `e7e8q` |
| castling | king move | `e1g1`, `e1c1` |
| null | literal | `0000` |

---

## OUTPUT lines (engine → GUI)

### Response to `uci`

Order: `id name`, `id author`, blank line, `option …` (one per line), `uciok`.

| Field | Format |
|---|---|
| name | `id name Stockfish 18` |
| author | `id author the Stockfish developers (see AUTHORS file)` |
| option | `option name <id> type <spin\|check\|string\|button\|combo> default <v> [min <n> max <n>]` |
| terminator | `uciok` |

Full SF18 option list (advertise only what you honor):

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

### `info` line (during `go`)

Token then value, space-separated, in this exact order. Omit any token not computed. Prefix `info`.

| Order | Token | Format | Notes |
|---|---|---|---|
| 1 | depth | `depth <n>` | full-width search depth |
| 2 | seldepth | `seldepth <n>` | selective depth |
| 3 | multipv | `multipv <n>` | rank index, 1-based |
| 4 | score | `score cp <n>` \| `score mate <n>` | cp = centipawns, side-to-move POV (+ = mover better); mate = mate in n moves, negative = being mated |
| 5 | wdl | `wdl <w> <d> <l>` | only if `UCI_ShowWDL true`; per-mille, sums 1000; immediately after score |
| 6 | bound | `lowerbound` \| `upperbound` | only on aspiration fail-high/low; flag, no value |
| 7 | nodes | `nodes <n>` | nodes searched |
| 8 | nps | `nps <n>` | nodes per second |
| 9 | hashfull | `hashfull <n>` | TT fill, per-mille |
| 10 | tbhits | `tbhits <n>` | tablebase hits |
| 11 | time | `time <ms>` | elapsed milliseconds |
| 12 | pv | `pv <m> …` | principal variation; MUST be last token |

Also: `info string <free text>` — diagnostics, GUI ignores. Standalone line.

Captured examples:

```
info depth 12 seldepth 19 multipv 1 score cp 47 nodes 41065 nps 821300 hashfull 11 tbhits 0 time 50 pv e2e4 c7c5 b1c3 a7a6 g1f3
info depth 1 seldepth 2 multipv 1 score mate 1 nodes 20 nps 20000 hashfull 0 tbhits 0 time 1 pv a1a8
info depth 10 seldepth 12 multipv 1 score cp 69 wdl 235 763 2 nodes 4835 nps 805833 hashfull 1 tbhits 0 time 6 pv e2e4 e7e5 g1f3 b8c6
```

### `bestmove` (terminates every `go` except perft)

| Case | Format | Example |
|---|---|---|
| normal | `bestmove <m> [ponder <m>]` | `bestmove e2e4 ponder c7c5` |
| no legal move | `bestmove (none)` | `bestmove (none)` |

### `go perft <n>` (no `bestmove`)

Order: one line per root move, blank line, total.

| Order | Format | Example |
|---|---|---|
| per move | `<m>: <count>` | `a2a3: 20` |
| separator | (blank line) | |
| total | `Nodes searched: <total>` | `Nodes searched: 400` |

### Debug dumps (non-UCI, optional)

| Command | Output lines |
|---|---|
| `d` | board grid, then `Fen: <FEN>`, `Key: <hex>`, `Checkers: <squares>` |
| `eval` | `NNUE evaluation        +0.12 (white side)` then `Final evaluation       +0.15 (white side) [with scaled NNUE, ...]` |

`d` board grid format:

```
 +---+---+---+---+---+---+---+---+
 | r | n | b | q | k | b | n | r | 8
 +---+---+---+---+---+---+---+---+
 …
   a   b   c   d   e   f   g   h
```

---

## Mandatory subset (must implement)

| Input | Output |
|---|---|
| `uci` | `id name`, `id author`, `option …`, `uciok` |
| `isready` | `readyok` |
| `ucinewgame` | none |
| `setoption …` | none |
| `position …` | none |
| `go …` | `info …`, `bestmove <m> [ponder <m>]` |
| `stop` | `bestmove <m> [ponder <m>]` |
| `quit` | exit |

Optional: `ponder`, `ponderhit`, `perft`, `d`, `eval`, `wdl`, MultiPV.

---

## Not reproducible on this hardware (standard UCI — include if implementing)

| Field | Format | Why absent |
|---|---|---|
| currmove | `currmove <m> currmovenumber <n>` | SF emits only when one root move searches >~3s; binary runs >1M nps, never fired |
| bound | `lowerbound` / `upperbound` | aspiration fail-high/low; short searches didn't trigger |
