from pathlib import Path
from collections.abc import Callable
from typing import Protocol

import chess
import torch

from chess_engine.features import boards_to_tensor
from chess_engine.models.nee import NeuralEvaluationNet
from chess_engine.models.policy import MovePolicyNet
from chess_engine.move_encoding import score_move_from_parts

class BestMoveOracle(Protocol):

  def evaluate_and_move(self, board: chess.Board) -> tuple[int, chess.Move]:
    ...

class StockfishMoveSelector:

  def __init__(self, oracle: BestMoveOracle):
    self.oracle = oracle

  def choose_move(self, board: chess.Board) -> chess.Move:
    _, move = self.oracle.evaluate_and_move(board)
    if move not in board.legal_moves:
      raise RuntimeError(f"Stockfish returned illegal move: {move.uci()}")
    return move

class NeuralMoveSelector:

  def __init__(
      self,
      checkpoint_path: str | Path,
      *,
      device_name: str | None = None,
      search_depth: int = 2,
  ):
    self.device = _select_device(device_name)
    self.search_depth = max(1, int(search_depth))
    self.eval_cache = {}
    self.model = NeuralEvaluationNet().to(self.device)
    state = torch.load(checkpoint_path, map_location=self.device)
    self.model.load_state_dict(state)
    self.model.eval()

  def choose_move(self, board: chess.Board) -> chess.Move:
    self.eval_cache = {}

    legal_moves = list(board.legal_moves)
    if not legal_moves:
      raise RuntimeError(f"no legal moves for {board.fen()}")
    if self.search_depth > 1:
      return choose_alphabeta_move(
          board,
          depth=self.search_depth,
          score_board=self._score_board,
          order_moves=self.score_moves,
      )
    scores = self.score_moves(board, legal_moves)
    best_index = max(range(len(legal_moves)), key=lambda index: scores[index])
    return legal_moves[best_index]

  def score_moves(
      self,
      board: chess.Board,
      legal_moves: list[chess.Move],
  ) -> list[float]:
    candidate_boards = []

    for move in legal_moves:
      candidate = board.copy(stack=False)
      candidate.push(move)
      candidate_boards.append(candidate)

    values = self._raw_board_values(candidate_boards)

    return [-float(value) for value in values]

  def _score_board(self, board: chess.Board, root_color: bool) -> float:
    score = _terminal_score(board, root_color)
    if score is not None:
      return score

    value = self._raw_board_values([board])[0]

    return value if board.turn == root_color else -value

  def _raw_board_values(self, boards: list[chess.Board]) -> list[float]:
    values: list[float | None] = [None] * len(boards)

    missing_boards = []
    missing_indexes = []

    for index, board in enumerate(boards):
      key = board.fen()

      if key in self.eval_cache:
        values[index] = self.eval_cache[key]
      else:
        missing_boards.append(board)
        missing_indexes.append(index)

    if missing_boards:
      tensors = boards_to_tensor(missing_boards,
                                 flip_if_black=True).to(self.device)

      with torch.inference_mode():
        model_values = self.model(tensors).view(-1).detach().cpu().tolist()

      for index, board, value in zip(missing_indexes, missing_boards,
                                     model_values):
        value = float(value)
        self.eval_cache[board.fen()] = value
        values[index] = value

    return [float(value) for value in values]

class PolicyMoveSelector:

  def __init__(self,
               checkpoint_path: str | Path,
               *,
               device_name: str | None = None):
    self.device = _select_device(device_name)
    self.model = MovePolicyNet().to(self.device)
    self.model.load_state_dict(
        torch.load(checkpoint_path, map_location=self.device))
    self.model.eval()

  def choose_move(self, board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
      raise RuntimeError(f"no legal moves for {board.fen()}")
    scores = self.score_moves(board, legal_moves)
    best_index = max(range(len(legal_moves)), key=lambda index: scores[index])
    return legal_moves[best_index]

  def score_moves(self, board: chess.Board,
                  legal_moves: list[chess.Move]) -> list[float]:
    tensors = boards_to_tensor([board], flip_if_black=True).to(self.device)
    with torch.no_grad():
      from_logits, to_logits, promo_logits = self.model(tensors)
    from_scores = from_logits.view(-1).detach().cpu().tolist()
    to_scores = to_logits.view(-1).detach().cpu().tolist()
    promo_scores = promo_logits.view(-1).detach().cpu().tolist()
    return [
        score_move_from_parts(move, board, from_scores, to_scores,
                              promo_scores) for move in legal_moves
    ]

ScoreBoard = Callable[[chess.Board, bool], float]
OrderMoves = Callable[[chess.Board, list[chess.Move]], list[float]]

def choose_minimax_move(
    board: chess.Board,
    *,
    depth: int,
    score_board: ScoreBoard,
) -> chess.Move:
  legal_moves = list(board.legal_moves)
  if not legal_moves:
    raise RuntimeError(f"no legal moves for {board.fen()}")
  root_color = board.turn
  scored = []
  for move in legal_moves:
    board.push(move)
    scored.append(
        (move, _minimax(board, max(0, depth - 1), root_color, score_board)))
    board.pop()
  return max(scored, key=lambda item: item[1])[0]

def _minimax(
    board: chess.Board,
    depth: int,
    root_color: bool,
    score_board: ScoreBoard,
) -> float:
  terminal = _terminal_score(board, root_color)
  if terminal is not None:
    return terminal
  if depth <= 0:
    return score_board(board, root_color)

  legal_moves = list(board.legal_moves)
  if board.turn == root_color:
    best = float("-inf")
    for move in legal_moves:
      board.push(move)
      best = max(best, _minimax(board, depth - 1, root_color, score_board))
      board.pop()
    return best

  best = float("inf")
  for move in legal_moves:
    board.push(move)
    best = min(best, _minimax(board, depth - 1, root_color, score_board))
    board.pop()
  return best

def _terminal_score(board: chess.Board, root_color: bool) -> float | None:
  if board.is_checkmate():
    return -10000.0 if board.turn == root_color else 10000.0
  if board.is_game_over(claim_draw=True):
    return 0.0
  return None

def _select_device(device_name: str | None) -> torch.device:
  if device_name:
    return torch.device(device_name)
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def choose_alphabeta_move(
    board: chess.Board,
    *,
    depth: int,
    score_board: ScoreBoard,
    order_moves: OrderMoves | None = None,
) -> chess.Move:
  legal_moves = list(board.legal_moves)

  if not legal_moves:
    raise RuntimeError(f"no legal moves for {board.fen()}")

  root_color = board.turn
  alpha = float("-inf")
  beta = float("inf")

  # Root move ordering helps alpha-beta prune earlier.
  # score_moves returns values good for the side to move,
  # and at root the side to move is root_color.
  if order_moves is not None:
    legal_moves = _ordered_moves(board, legal_moves, order_moves)

  best_move = legal_moves[0]
  best_score = float("-inf")

  for move in legal_moves:
    board.push(move)
    score = _alphabeta(
        board,
        max(0, depth - 1),
        root_color,
        score_board,
        alpha,
        beta,
        order_moves,
    )
    board.pop()

    if score > best_score:
      best_score = score
      best_move = move

    alpha = max(alpha, best_score)

  return best_move

def _alphabeta(
    board: chess.Board,
    depth: int,
    root_color: bool,
    score_board: ScoreBoard,
    alpha: float,
    beta: float,
    order_moves: OrderMoves | None = None,
) -> float:
  terminal = _terminal_score(board, root_color)

  if terminal is not None:
    return terminal

  if depth <= 0:
    return score_board(board, root_color)

  legal_moves = list(board.legal_moves)

  # Do not order at depth 1, because that would evaluate the same leaves twice.
  # Ordering is mainly useful when there is still deeper search below.
  if order_moves is not None and depth > 1:
    legal_moves = _ordered_moves(board, legal_moves, order_moves)

  if board.turn == root_color:
    value = float("-inf")

    for move in legal_moves:
      board.push(move)
      value = max(
          value,
          _alphabeta(
              board,
              depth - 1,
              root_color,
              score_board,
              alpha,
              beta,
              order_moves,
          ),
      )
      board.pop()

      alpha = max(alpha, value)

      if alpha >= beta:
        break

    return value

  value = float("inf")

  for move in legal_moves:
    board.push(move)
    value = min(
        value,
        _alphabeta(
            board,
            depth - 1,
            root_color,
            score_board,
            alpha,
            beta,
            order_moves,
        ),
    )
    board.pop()

    beta = min(beta, value)

    if alpha >= beta:
      break

  return value

def _ordered_moves(
    board: chess.Board,
    legal_moves: list[chess.Move],
    order_moves: OrderMoves,
) -> list[chess.Move]:
  scores = order_moves(board, legal_moves)

  return [
      move for move, _ in sorted(
          zip(legal_moves, scores),
          key=lambda item: item[1],
          reverse=True,
      )
  ]
