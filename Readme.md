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

All measurements on **Apple Silicon M1** (MacBook Pro, single-threaded), compiled with `gcc -O3 -march=native`.

### Nodes Per Second (NPS)

| Depth | NNUE (768→256→32→1) | Material-Only |
|-------|---------------------|---------------|
| 1     | 20 nodes, 3.4 ms, **5,963 NPS** | 20 nodes, 2.4 ms, **8,372 NPS** |
| 2     | 368 nodes, 55 ms, **6,686 NPS** | 368 nodes, 42 ms, **8,770 NPS** |
| 3     | 1,531 nodes, 181 ms, **8,453 NPS** | 1,531 nodes, 173 ms, **8,876 NPS** |
| 4     | 3,247 nodes, 339 ms, **9,583 NPS** | 3,247 nodes, 338 ms, **9,613 NPS** |
| 5     | 105k nodes, 11.9 s, **8,878 NPS** | 105k nodes, 11.2 s, **9,459 NPS** |
| 6     | 391k nodes, 191.7 s, **2,041 NPS** ⚠️ | — |

> ⚠️ At depth 6, thermal throttling reduces NPS significantly. The engine has **no transposition table**, so search tree size grows exponentially. Depths 1–5 are the practical range.

### Self-Play Speed (Depth 4, NNUE ON)

| Metric | Value |
|--------|-------|
| Average time per move | **650 ms** |
| Average nodes per move | **6,541** |
| Average NPS | **~10,000** |
| Total time (30 moves) | **19.5 seconds** |

### Key Takeaways

- **~10,000 NPS** on Apple M1 (single core) — respectable for a custom engine without Magic Bitboards or transposition tables
- **NNUE eval adds negligible overhead** (~5% slower than material-only) thanks to sparse input optimization (only multiplies non-zero input activations)
- The engine performs best at **depths 3–5** (sub-second to ~12 seconds per move)
- **No transposition table** means significant redundant work; adding one would increase effective search depth by 1–2 ply

### What This Engine Is (and Isn't)

NeuralBit Chess is an **educational chess engine** that demonstrates:
- ✅ Bitboard move generation with full legal move filtering
- ✅ Alpha-beta search with Null Move Pruning and Late Move Reductions
- ✅ NNUE evaluation with efficient sparse inference in C
- ✅ Complete self-play training pipeline (C++ LibTorch + Python)

It is **not** competitive with top engines like Stockfish (which achieves 1–2 million NPS on the same hardware, with iterative deepening, transposition tables, advanced move ordering, and quiescence search). This engine is a **learning project** that shows solid understanding of modern chess programming techniques — it will beat casual human players but is not designed for engine tournaments.

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
