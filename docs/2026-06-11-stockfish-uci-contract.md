Parent: none

# Stockfish UCI I/O contract (engine build reference)

## Status

Complete. Captured empirically from the bundled binary on 2026-06-11:
`src/llm/runtime/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe` → **Stockfish 18**.
Every line below is real engine output, not recalled. Three items could not be
forced on this hardware and are flagged as unverified.

## Scope

The exact stdin/stdout contract a custom engine must implement to be a drop-in
replacement for Stockfish under the UCI protocol: transport, every input command,
every output line, move encoding, score semantics, and the minimal mandatory
subset.

---

## Transport

- Line-based. Commands arrive on **stdin**, responses go to **stdout**, each line
  `\n`-terminated ASCII. Flush per line (do not buffer — GUIs read synchronously).
- On process launch, before any input, the engine prints one banner line:
  ```
  Stockfish 18 by the Stockfish developers (see AUTHORS file)
  ```
- `stderr` is not part of the protocol.
- `go` is asynchronous: the engine keeps reading stdin (`stop`, `isready`, `quit`)
  while a search runs. A `quit` sent mid-search aborts the search immediately.

## INPUT commands (GUI → engine)

| Command | Args | Engine response |
|---|---|---|
| `uci` | — | `id` lines + all `option` lines + `uciok` |
| `isready` | — | `readyok` (must reply **anytime**, even mid-search) |
| `ucinewgame` | — | **no output** |
| `setoption name <id> [value <v>]` | — | **no ack** (some emit an `info string`, e.g. `info string Using 2 threads`) |
| `position startpos [moves <m1> <m2>…]` | — | **no output** |
| `position fen <FEN> [moves …]` | — | **no output** |
| `go <limits>` | see below | stream `info` lines, terminate with `bestmove` |
| `stop` | — | abort search now, emit `bestmove` |
| `ponderhit` | — | leave ponder, search normally, eventually `bestmove` |
| `quit` | — | exit process |
| `d` | — | ASCII board (debug, non-UCI) |
| `eval` | — | eval breakdown (debug, non-UCI) |
| *unknown* | — | `Unknown command: '<x>'. Type help for more information.` (keeps running) |

`go` limit tokens (combinable): `depth N`, `movetime <ms>`, `nodes N`,
`wtime <ms> btime <ms> winc <ms> binc <ms>`, `movestogo N`, `infinite`,
`mate N`, `ponder`, `searchmoves <m1> <m2>…`, `perft N`.

## OUTPUT lines (engine → GUI)

### `uci` handshake

Head + tail verbatim; option lines carry `id` / `type` / `default` / `min` / `max`.
`type` ∈ `spin | check | string | button | combo`.

```
id name Stockfish 18
id author the Stockfish developers (see AUTHORS file)

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
uciok
```

A custom engine only needs to advertise the options it actually honors.

### `info` line

Fields in this observed order (omit any you do not compute):

```
info depth 12 seldepth 19 multipv 1 score cp 47 nodes 41065 nps 821300 hashfull 11 tbhits 0 time 50 pv e2e4 c7c5 b1c3 a7a6 g1f3
```

- `score cp N` — centipawns, **side-to-move POV** (`+` favors the mover).
- `score mate N` — mate in N moves; **negative** = being mated. PV ends at mate:
  ```
  info depth 1 seldepth 2 multipv 1 score mate 1 nodes 20 nps 20000 hashfull 0 tbhits 0 time 1 pv a1a8
  ```
- `wdl W D L` — only when `UCI_ShowWDL true`; inserted **right after score**,
  per-mille, sums to 1000:
  ```
  info depth 10 seldepth 12 multipv 1 score cp 69 wdl 235 763 2 nodes 4835 nps 805833 hashfull 1 tbhits 0 time 6 pv e2e4 e7e5 g1f3 b8c6
  ```
- MultiPV: one line per ranked move, field `multipv 1|2|3…`:
  ```
  info depth 8 seldepth 11 multipv 1 score cp 50 ... pv e2e4 e7e5 g1f3 g8f6 f3e5 f6e4 f1c4
  info depth 8 seldepth 13 multipv 2 score cp 22 ... pv d2d4 d7d5 c2c4 e7e6 g1f3 g8f6 b1c3 d5c4
  info depth 8 seldepth 12 multipv 3 score cp 22 ... pv c2c4 e7e5 b1c3 b8c6 g1f3 f8b4
  ```
- `info string <free text>` — human-readable messages (NNUE load, thread count).
  GUI ignores these; safe to use for diagnostics.

### `bestmove`

Terminates every `go` except `perft`:

```
bestmove e2e4 ponder c7c5
```

`ponder <move>` is optional. When there is no legal move: `bestmove (none)`.

### `go perft N`

Not a search. Lists each root move with its child count, a blank line, then the
total. No `bestmove`:

```
a2a3: 20
b2b3: 20
...
g1h3: 20

Nodes searched: 400
```

## Move encoding

Long algebraic: `<from><to>[promo]`.
- `e2e4` — normal move.
- `e7e8q` — promotion; promo piece lowercase `q | r | b | n`.
- `e1g1` / `e1c1` — castling expressed as the **king move**.
- `0000` — null move. `(none)` — no legal move available.

## Debug commands (optional, non-UCI)

`d` →
```
 +---+---+---+---+---+---+---+---+
 | r | n | b | q | k | b | n | r | 8
 +---+---+---+---+---+---+---+---+
 ...
   a   b   c   d   e   f   g   h

Fen: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
Key: B46022469E3DD31B
Checkers:
```

`eval` →
```
NNUE evaluation        +0.12 (white side)
Final evaluation       +0.15 (white side) [with scaled NNUE, ...]
```

## Not forced empirically (standard UCI — document, unverified this run)

- `currmove <m> currmovenumber N` — printed only when one root move's search
  exceeds ~3s. The bundled binary runs >1M nps, never spent that long on a single
  move, so the field never appeared in these captures.
- `lowerbound` / `upperbound` — emitted on aspiration-window fail-high/fail-low;
  short searches did not trigger one.
- `refutation` / `currline` — gated behind `debug` / `UCI_AnalyseMode`.

## Minimal viable engine

Handle exactly:

| Command | Required reply |
|---|---|
| `uci` | `id` ×2 + options + `uciok` |
| `isready` | `readyok` |
| `ucinewgame` | (nothing) |
| `setoption` | (nothing) |
| `position` | (nothing) |
| `go` | `info` lines* + `bestmove` |
| `stop` | `bestmove` |
| `quit` | exit |

Everything else (`ponder`, `perft`, `d`, `eval`, WDL, MultiPV) is optional.

## Evidence

All output above was captured by piping commands into the binary and reading
stdout. `quit` must be delayed (e.g. `sleep`) after `go` or it aborts the search
before `info`/`bestmove` flush. Representative commands:

```bash
SF="src/llm/runtime/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"

# handshake
printf 'uci\nquit\n' | "$SF"

# search with full info stream (delay quit so search completes)
( echo "position startpos"; echo "go depth 12"; sleep 3; echo quit ) | "$SF"

# mate score
( echo "position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"; echo "go depth 6"; sleep 1; echo quit ) | "$SF"

# MultiPV, perft, eval, board, WDL, unknown-command, setoption acks
( echo "setoption name MultiPV value 3"; echo "position startpos"; echo "go depth 8"; sleep 1; echo quit ) | "$SF"
( echo "position startpos"; echo "go perft 2"; sleep 1; echo quit ) | "$SF"
( echo "setoption name UCI_ShowWDL value true"; echo "position startpos"; echo "go depth 10"; sleep 1; echo quit ) | "$SF"
```

Outcomes matched the line formats documented above. `setoption` and `ucinewgame`
produced no acknowledgement; `isready` always returned `readyok`; an unknown
command returned `Unknown command: '<x>'. Type help for more information.` and the
engine kept running.

## Next

1. Implement the minimal viable command set first; verify byte-for-byte against the
   capture commands in **Evidence**.
2. Add `info` streaming (depth/score/nodes/nps/time/pv) — GUIs and the agent loop
   depend on `bestmove` plus at least one `score` line.
3. Add `perft` for move-generation correctness testing against known node counts
   (startpos perft 4 = 197281).
4. Defer `ponder`, MultiPV, WDL, Syzygy until the core loop is proven.
