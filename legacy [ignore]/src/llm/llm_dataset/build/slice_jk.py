"""Author slices J (plain chat) and K (adversarial routing negatives).

The human J/K slices collapsed to ~15 distinct conversations each after exact
de-dup, so we regenerate diverse coverage from phrasing pools.

  J   -> direct reply, NO tool.
  K-1 -> chess-flavored *knowledge* (abstract) -> ask_chessbot, then narrate.
  K-2 -> off-topic message containing chess words -> direct reply, NO tool.
"""
from __future__ import annotations

import random

from llm_training.system_prompt import SYSTEM_PROMPT

# ---- J: plain chat (user, warm direct reply) -----------------------------
J_PAIRS = [
    ("Hey there!", "Hey! Great to see you at the board - ready when you are."),
    ("Good morning!", "Good morning! Hope you're feeling sharp - shall we play some chess?"),
    ("How are you?", "Doing great and ready to help with your game! How are you?"),
    ("Do you like chess?", "I love it - every position is a fresh little puzzle. What about you?"),
    ("What can you do?", "I can play your moves, evaluate the position, suggest plans, review your moves, and chat chess with you."),
    ("Thanks for the help!", "Anytime - that's what I'm here for! Enjoy the game."),
    ("I'm not very good at chess.", "No worries at all - everyone starts somewhere, and I'm happy to help you improve."),
    ("This is fun.", "So glad you're enjoying it! Chess is the best kind of fun."),
    ("I appreciate your advice.", "That means a lot - happy to help whenever you need it!"),
    ("Are you a real chess engine?", "I'm your coach! A real engine does the heavy calculation in the background, and I help you make sense of it."),
    ("Who are you?", "I'm your friendly chess coach - here to help you play, plan, and improve."),
    ("I'm ready to play.", "Love the energy - make your move whenever you're set!"),
    ("Good night.", "Good night! Come back for a game anytime."),
    ("You're pretty helpful.", "Aw, thank you - I really enjoy helping you play better!"),
    ("Let's have a good game.", "Absolutely - here's to a fun, fighting game!"),
    ("I'm a bit nervous.", "Totally normal - take a breath, trust your instincts, and we'll take it one move at a time."),
    ("What's your favorite opening?", "I have a soft spot for the Italian Game - classy and full of ideas! Do you have a favorite?"),
    ("Do you ever lose?", "I'm just the coach, but everybody loses sometimes - it's how we learn!"),
    ("Tell me a chess fact.", "Here's one: the number of possible chess games is famously larger than the number of atoms in the observable universe!"),
    ("I think I'll win this one.", "Love the confidence - let's go make it happen!"),
]
J_OPENERS = ["", "Hi! ", "Hello! ", "Hey! ", "Oh, ", "Sure - "]

# ---- K-2: off-topic with chess words (direct reply, NO tool) --------------
K2_PAIRS = [
    ("I bought a queen-sized bed today.", "Nice upgrade - a queen-sized bed sounds comfy! Let me know if you want to play some chess too."),
    ("The king of pop is Michael Jackson.", "True! Though the only king I can help with is the one on the board."),
    ("My horse is named Rook.", "Ha, what a great name for a horse! On the board my rooks move in straight lines, sadly no galloping."),
    ("Checkmate, that's a deal!", "Sounds like a win-win - though if you want a real checkmate, I'm your coach!"),
    ("I'm going to the gym to work on my pawns... I mean abs.", "Ha! Train those abs - I'll keep an eye on your pawns over here."),
    ("She's the queen of the office.", "Sounds like she runs the place! Speaking of queens, want to deploy yours on the board?"),
    ("That movie was a real stalemate of a plot.", "Ha, a plot going nowhere - at least a real stalemate ends the game! Fancy a game?"),
    ("My knight in shining armor showed up.", "How romantic! My knights only shine when they fork a queen, though."),
    ("We castled into our new house last week.", "Congrats on the new place - castling to safety, just like in chess!"),
    ("The bishop gave a lovely sermon on Sunday.", "That's lovely! On the board my bishops prefer long diagonals to long sermons."),
    ("I'm developing a new app.", "Exciting! I mostly develop knights and bishops, but I wish you luck with the app."),
    ("Traffic was a total gridlock today.", "Ugh, gridlock is the worst - at least on the board I can help you break through!"),
    ("King Charles is on the news.", "Royalty in the headlines! The only crowns I track are the kings on your board."),
    ("I'll trade you my sandwich for your chips.", "Ha, I'm better at trading bishops than snacks - but that sounds like a fair deal!"),
    ("My favorite band is Queen.", "Great taste - rock royalty! Want to bring your own queen out to play?"),
]
K2_OPENERS = ["", "Ha! ", "Nice - ", "Oh, "]

# ---- K-1: abstract chess knowledge -> ask_chessbot ------------------------
K1_QA = [
    ("what is a knight worth in general", "A knight is worth about 3 points, the same as a bishop in most situations."),
    ("is the queen the strongest piece", "Yes - the queen is the most powerful piece, worth roughly 9 points and combining rook and bishop movement."),
    ("how many squares does a bishop control on average", "A bishop controls about 7 squares on average, more when placed in the open center."),
    ("what is a rook worth", "A rook is worth about 5 points, making it a major piece behind only the queen."),
    ("what is a fork in chess", "A fork is a tactic where one piece attacks two or more enemy pieces at the same time."),
    ("what is a pin", "A pin is when a piece can't move because doing so would expose a more valuable piece behind it."),
    ("why is castling good", "Castling tucks your king into safety and brings a rook toward the center in a single move."),
    ("what is a passed pawn", "A passed pawn has no enemy pawns able to stop it from promoting on its file or adjacent files."),
    ("what does it mean to control the center", "Controlling the center gives your pieces more mobility and makes it easier to attack on either side."),
    ("what is the value of a pawn", "A pawn is the baseline unit, worth about 1 point."),
    ("what is a discovered attack", "A discovered attack happens when moving one piece reveals an attack from another piece behind it."),
    ("what is zugzwang", "Zugzwang is a situation where any move a player makes worsens their position."),
    ("what is a skewer", "A skewer attacks a valuable piece, forcing it to move and exposing a lesser piece behind it."),
    ("are two bishops better than two knights", "Often yes - the bishop pair tends to be strong in open positions with play on both sides."),
    ("what is the bishop pair", "The bishop pair is the advantage of having both bishops, which cover squares of every color together."),
    ("what is a back-rank mate", "A back-rank mate delivers checkmate along the king's home rank when its own pawns block escape."),
]
K1_PHRASE = ["{q}?", "Can you tell me {q}?", "I was wondering, {q}?", "Quick question - {q}?",
             "In general, {q}?", "Just curious: {q}?", "Hey, {q}?"]
K1_INTROS = ["Good question! ", "Sure thing - ", "Happy to explain: ", "Great one - ", ""]


def _rec(uid, sl, messages, note):
    return {"id": uid, "slice": sl,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "validated": True, "notes": note}


def generate_j(n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        user, reply = rng.choice(J_PAIRS)
        out.append(_rec(f"ex_J_{i:04d}", "J",
                        [{"role": "user", "content": user},
                         {"role": "assistant", "content": rng.choice(J_OPENERS) + reply}],
                        "plain chat, no tool"))
    return out


def generate_k(n1: int, n2: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n1):
        q, ans = rng.choice(K1_QA)
        user = rng.choice(K1_PHRASE).format(q=q).capitalize()
        out.append(_rec(f"ex_K1_{i:04d}", "K",
                        [{"role": "user", "content": user},
                         {"role": "assistant", "content": f"<tool>ask_chessbot query={q}</tool>"},
                         {"role": "tool", "content": ans},
                         {"role": "assistant", "content": rng.choice(K1_INTROS) + ans}],
                        "K-1 chess-flavored knowledge -> ask_chessbot"))
    for i in range(n2):
        user, reply = rng.choice(K2_PAIRS)
        out.append(_rec(f"ex_K2_{i:04d}", "K",
                        [{"role": "user", "content": user},
                         {"role": "assistant", "content": rng.choice(K2_OPENERS) + reply}],
                        "K-2 off-topic with chess words, no tool"))
    return out
