# ♟️ NeuralBit Chess: HPC C-Engine & Deep RL PyTorch

[![C](https://img.shields.io/badge/C-17-blue.svg)](https://en.cppreference.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

Welcome to **NeuralBit Chess**, an engineering project combining the raw speed of a low-level chess engine (Native C / Bitboards) with a neural network trained via Reinforcement Learning (Deep RL / Actor-Critic architecture inspired by AlphaZero).

This project is divided into two main parts:

1. **The Core Engine (`engine.c`)**: A custom chess engine built from scratch in C using bitboards, with a built-in NNUE evaluator and alpha-beta search featuring Null Move Pruning and Late Move Reductions.
2. **The Artificial Intelligence Loop**: A complete autonomous training pipeline (Self-Play in C++ with LibTorch → Policy Gradient Optimization in Python).

---

## 📊 Benchmarks

All measurements on **Apple Silicon M1** (MacBook Pro, single-threaded), compiled with `gcc -O3 -march=native`. Two modes are benchmarked:

- **Perft** — pure legal move generation + make/unmake, no evaluation, no pruning. Measures raw movegen throughput.
- **Search** — full alpha-beta search with NMP + LMR, measuring practical playing speed.

### Perft Speed (Move Generation)

| Position | Depth | Nodes | Time | **NPS** |
|----------|-------|-------|------|---------|
| Start (20 moves) | 4 | 197,281 | 3.8 ms | **51.9M** |
| Start (20 moves) | 5 | 4,865,351 | 93.9 ms | **51.8M** |
| Endgame (K+R vs K) | 5 | 118,023 | 3.6 ms | **32.5M** |
| Endgame (K+R vs K) | 6 | 692,754 | 29.2 ms | **23.7M** |

> The engine achieves **~52 million NPS** in perft mode on the starting position, demonstrating efficient bitboard move generation with `__builtin_ctzll` and copy-make-unmake on the stack.

### Search Speed (Alpha-Beta + NNUE)

#### Opening Position (all 32 pieces)

| Depth | NNUE (768→256→32→1) | Material-Only |
|-------|---------------------|---------------|
| 1 | 20 nodes, 2.4 ms, **8,271 NPS** | 20 nodes, 2.5 ms, **8,061 NPS** |
| 2 | 368 nodes, 41.6 ms, **8,849 NPS** | 368 nodes, 41.6 ms, **8,840 NPS** |
| 3 | 1,531 nodes, 169 ms, **9,069 NPS** | 1,531 nodes, 173 ms, **8,854 NPS** |
| 4 | 3,247 nodes, 330 ms, **9,834 NPS** | 3,247 nodes, 339 ms, **9,574 NPS** |
| 5 | 105k nodes, 11.3 s, **9,376 NPS** | 105k nodes, 11.3 s, **9,359 NPS** |

#### Endgame (K+R vs K, 3 pieces)

| Depth | NNUE | Material-Only |
|-------|------|---------------|
| 3 | 622 nodes, 53 ms, **11,712 NPS** | 622 nodes, 48 ms, **13,061 NPS** |
| 4 | 596 nodes, 32 ms, **18,494 NPS** | 596 nodes, 29 ms, **20,487 NPS** |
| 5 | 3,978 nodes, 287 ms, **13,842 NPS** | 3,978 nodes, 283 ms, **14,045 NPS** |

### Self-Play Speed (Depth 4, Starting Position, NNUE ON)

| Metric | Value |
|--------|-------|
| Average time per move | **650 ms** |
| Average nodes per move | **6,541** |
| Average NPS | **~10,000** |
| 30 moves completed in | **19.5 seconds** |

### Key Takeaways

- **Move generation is fast** — ~52 million NPS in perft mode on the starting position. This is the engine's strength and confirms the bitboard implementation is efficient.
- **NNUE adds ~5% overhead** vs material-only evaluation, thanks to the sparse-input optimization (only multiplies non-zero input activations, avoiding 196,000 useless multiplications per evaluation).
- **NPS scales with position** — endgames are faster (fewer pieces → faster eval and movegen). Endgame search reaches 20,000 NPS vs ~9,500 in the opening.
- **No transposition table** means significant redundant work; adding one would increase effective search depth by 1–2 ply.
- The engine performs best at **depths 3–5** (sub-second to ~12 seconds per move in the opening).

### What This Engine Is (and Isn't)

NeuralBit Chess is an **educational chess engine** that demonstrates:
- ✅ Bitboard move generation with full legal move filtering — **52M NPS perft**
- ✅ Alpha-beta search with Null Move Pruning and Late Move Reductions
- ✅ NNUE evaluation with efficient sparse inference in C — only ~5% overhead
- ✅ Complete self-play training pipeline (C++ LibTorch + Python)

It is **not** competitive with top engines like Stockfish (which achieves 1–2 million NPS **in search mode** on the same hardware, with iterative deepening, transposition tables, magic bitboards, advanced move ordering, and quiescence search). This engine is a **learning project** that shows solid understanding of modern chess programming techniques — it will beat casual human players at depth 4–5 but is not designed for engine tournaments.

---

## 🚀 Part 1: The Native C Engine (`engine.c` API)

The `engine.c` file is the beating heart of the project. It uses no external dependencies and relies entirely on bit manipulation (Bitboards).

### Quick Compile & Play

```bash
# Compile and play against the AI (material-only eval)
gcc -O3 -march=native -o play_chess play.c
./play_chess
```

### Architecture

| Component | Implementation |
|-----------|---------------|
| **Board** | 12 × 64-bit bitboards (one per piece type × color) |
| **Move Generation** | Bitwise shifts + `__builtin_ctzll`, legal filtering |
| **Evaluation** | NNUE (768→256→32→1) + material fallback |
| **Search** | Alpha-Beta + Null Move Pruning + Late Move Reductions |
| **Sliding Pieces** | Fallback ray-based (no Magic Bitboards) |
| **Transposition Table** | Not implemented |

### Use `engine.c` as an API

To use the engine in your own projects, simply include the file (ensure the original `main` function is renamed or commented out):

```C
#include "engine.c"
```

#### Main Structures

- **`Board`**: The state of the chessboard. Contains 12 `uint64_t` (Bitboards) for pieces, castling rights (`castling_rights`), en passant target square (`en_passant_square`), and the 50-move rule clock.
- **`MoveList`**: Contains an array of 32-bit integers representing legal moves.
- **Move Encoding**: A move is a simple `uint32_t`. Use the macros `GET_FROM(move)`, `GET_TO(move)`, `GET_PIECE(move)`, `GET_PROMOTION(move)`, and `GET_FLAGS(move)` to decode it.

#### Essential Functions

- `void init_leapers_and_masks()`: **Mandatory at startup.** Precalculates attack masks (Knights, Kings, Rays).
- `void generate_moves(Board *b, MoveList *list, int color)`: Populates the `MoveList` with all pseudo-legal moves.
- `void make_move(Board *b, uint32_t move, int color)`: Applies a move to the `Board` (updates bitboards incrementally).
- `int check_game_over(Board *b, int color)`: Returns `0` (Continue), `1` (Checkmate), or `2` (Stalemate).
- `uint32_t search_best_move(Board *b, int depth, int color, int eval_type)`: The optimized Alpha-Beta algorithm (with Null Move Pruning and Late Move Reductions) that returns the calculated best move.

---

## 🧠 Part 2: The Deep Reinforcement Learning Pipeline

This project implements an autonomous learning loop. The AI plays against itself in C++ to generate experience, then learns from its victories and defeats in Python using the **Policy Gradient** algorithm.

### Prerequisites

- Python 3.8+ (`pip install torch pandas matplotlib numpy`)
- CMake (3.14+)
- **LibTorch** (The C++ API for PyTorch, download from the official PyTorch website and extract it on your machine).

_(macOS Users: If LibTorch is blocked by Gatekeeper, run `sudo xattr -r -d com.apple.quarantine /path/to/libtorch`)_.

### The Step-by-Step Training Loop

#### Step 1: Initialize the Actor-Critic Architecture

Create the `.pt` model (TorchScript) readable by C++.

```bash
python rl_model.py
```

_Generates `actor_critic_model.pt`._

#### Step 2: Compile the C++ Executables (HPC)

The self-play factory (`selfplay.cpp`) and the evaluation arena (`arena.cpp`) must be compiled by linking LibTorch.

```bash
mkdir build
cd build

# Replace the path with your absolute path to libtorch

cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```

#### Step 3: Self-Play (Experience Generation)

The AI (The Actor) plays dozens of games against itself asynchronously and saves its decisions.

```bash
cd build
cp ../actor_critic_model.pt .
./selfplay
```

_Generates `rl_dataset.csv`._

#### Step 4: Mutation (RL Training)

The AI updates its neural networks: the probability of moves that led to a win increases, while others decrease.

```bash

# From the root directory

python train_rl.py
```

_Updates `actor_critic_model.pt` in the `build/` folder and generates a `loss_curve.png` plot._

**Repeat steps 3 and 4** to continually improve the AI's skill level (Continuous Training).

#### Step 5: The Arena (Strength Test)

Pit your newly trained AI (White) against the classic material Alpha-Beta engine (Black).

```bash
cd build
./arena
```

**Note** that it is also possible to begin by training a first neural network to predict the value of a valuation function of the boards
and then use this neural network and its weights for the Actor-Critic training so it doesn't start from scratch.
For that purpose you have the two code `generate_data.c` and `train.py`.

---

## 📁 File Overview

| File | Description |
|------|-------------|
| `engine.c` | Core chess engine (bitboards, NNUE, alpha-beta) |
| `play.c` | Interactive terminal play against the AI |
| `arena.cpp` | RL vs Classical tournament (requires LibTorch) |
| `selfplay.cpp` | Self-play data generation (requires LibTorch) |
| `train.py` | Supervised NNUE training (board → evaluation) |
| `train_rl.py` | Reinforcement Learning training (Policy Gradient) |
| `rl_model.py` | Actor-Critic model definition (TorchScript export) |
| `generate_data.c` | Generate training data from engine self-play |
| `export_weights.py` | Export PyTorch weights to `nnue_weights.bin` |
| `nnue_weights.bin` | Pre-trained NNUE weights (820 KB) |
| `CMakeLists.txt` | Build configuration for C++ components |

---

_Project built from scratch in C/C++ and Python._
