#!/usr/bin/env python3
"""
Discard Hold'em bot (heads-up, 3 hole cards, discard one after flop betting).

Single file, Python 3.11, stdlib + numpy only.  Protocol: one JSON object per
line on stdin, one JSON reply per line on stdout.  All diagnostics go to stderr.

Layout
  1. cards + vectorised bitmask hand evaluator (5..8 cards)
  2. equity engine (preflop / flop Monte Carlo, exact turn / river enumeration)
  3. opponent model (VPIP, PFR, AF, fold-to-bet, WTSD, showdown strength ...)
  4. bot: protocol handling, state tracking, time-bank management
  5. strategy: preflop, postflop, discard
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # no BLAS thread pool: faster start-up, no contention on a small sandbox
import sys
import json
import time
import random

import numpy as np

# ---------------------------------------------------------------------------
# 1. Cards and hand evaluator
# ---------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
_SUIT_ALIASES = {"♣": "c", "♦": "d", "♥": "h", "♠": "s"}
_RANK_WORDS = {"1": "A", "14": "A", "13": "K", "12": "Q", "11": "J", "10": "T"}


def parse_card(c):
    """Any reasonable card representation -> int 0..51 (rank * 4 + suit), or None."""
    try:
        if isinstance(c, bool):
            return None
        if isinstance(c, int):
            return c if 0 <= c < 52 else None
        if isinstance(c, dict):
            r = c.get("rank", c.get("r"))
            s = c.get("suit", c.get("s"))
            if r is None or s is None:
                return None
            c = str(r) + str(s)
        s = str(c).strip()
        for k, v in _SUIT_ALIASES.items():
            s = s.replace(k, v)
        s = s.replace("10", "T")
        if len(s) < 2:
            return None
        a, b = s[0], s[-1]
        mid = s[:-1]
        if mid in _RANK_WORDS:
            a = _RANK_WORDS[mid]
        if a.upper() in RANKS and b.lower() in SUITS:
            return RANKS.index(a.upper()) * 4 + SUITS.index(b.lower())
        if b.upper() in RANKS and a.lower() in SUITS:      # suit-first form, e.g. "sA"
            return RANKS.index(b.upper()) * 4 + SUITS.index(a.lower())
    except Exception:
        pass
    return None


def parse_cards(lst):
    """List (or compact string) of cards -> (ints, raw representations)."""
    ints, raws = [], []
    if lst is None:
        return ints, raws
    if isinstance(lst, str):
        t = lst.replace(",", " ").split()
        if len(t) == 1 and len(t[0]) > 3 and len(t[0]) % 2 == 0:
            t = [t[0][i:i + 2] for i in range(0, len(t[0]), 2)]
        lst = t
    if not isinstance(lst, (list, tuple)):
        return ints, raws
    for x in lst:
        v = parse_card(x)
        if v is not None and v not in ints:
            ints.append(v)
            raws.append(x)
    return ints, raws


NM = 8192
M13 = 0x1FFF
POP = np.zeros(NM, np.int64)        # popcount
HI = np.zeros(NM, np.int64)         # index of highest set bit
TOP1 = np.zeros(NM, np.int64)       # mask of the top-k set bits
TOP2 = np.zeros(NM, np.int64)
TOP3 = np.zeros(NM, np.int64)
TOP5 = np.zeros(NM, np.int64)
STRAIGHT = np.zeros(NM, np.int64)   # 0 = none, else (rank index of high card) + 1
FLUSHV = np.zeros(NM, np.int64)     # full hand value if this suit mask is a flush
SOUTS = np.zeros(NM, np.int64)      # number of ranks completing a straight

C1, C2, C3, C4, C5, C6, C7, C8 = [np.int64(i << 26) for i in range(1, 9)]
WHEEL = 0x100F


def _build_tables():
    pop = [0] * NM
    strt = [0] * NM
    for m in range(1, NM):
        bits = [i for i in range(12, -1, -1) if (m >> i) & 1]
        pop[m] = len(bits)
        HI[m] = bits[0]
        t = 0
        for j, bi in enumerate(bits[:5]):
            t |= 1 << bi
            if j == 0:
                TOP1[m] = t
            if j == 1:
                TOP2[m] = t
            if j == 2:
                TOP3[m] = t
        TOP5[m] = t
        if len(bits) < 2:
            TOP2[m] = m
        if len(bits) < 3:
            TOP3[m] = m
        s = 0
        for hi in range(12, 3, -1):
            w = 0x1F << (hi - 4)
            if (m & w) == w:
                s = hi + 1
                break
        if s == 0 and (m & WHEEL) == WHEEL:
            s = 4
        strt[m] = s
    for m in range(NM):
        POP[m] = pop[m]
        STRAIGHT[m] = strt[m]
        if pop[m] >= 5:
            FLUSHV[m] = (int(C8) | strt[m]) if strt[m] else (int(C5) | int(TOP5[m]))
        if strt[m] == 0:
            n = 0
            for r in range(13):
                if not (m >> r) & 1 and strt[m | (1 << r)]:
                    n += 1
            SOUTS[m] = n


_build_tables()

# One int64 per card: its rank bit placed in the 13-bit field of its suit.
CARD_KEY = np.array([(1 << (c >> 2)) << (13 * (c & 3)) for c in range(52)], dtype=np.int64)


def keys_of(cards):
    """Sum of card keys for an iterable of card ints (python int)."""
    k = 0
    for c in cards:
        k += int(CARD_KEY[c])
    return k


def eval_keys(key, draws=False):
    """
    Vectorised best-5 evaluation of card sets (5..8 distinct cards) given as
    summed CARD_KEYs.  Returns int64 values, larger = stronger:
    category << 26 | tiebreak fields (8 = straight flush ... 0 = high card).
    With draws=True also returns a draw bonus (flush draw / straight outs).
    """
    key = np.asarray(key, dtype=np.int64)
    a = key & M13
    b = (key >> 13) & M13
    c = (key >> 26) & M13
    d = (key >> 39) & M13
    ab = a & b
    cd = c & d
    aob = a | b
    cod = c | d
    anyr = aob | cod
    ge2 = ab | cd | (aob & cod)
    ge3 = (ab & cod) | (cd & aob)
    quad = ab & cd

    v = np.where(ge2 != 0,
                 C1 | (HI[ge2] << 13) | TOP3[anyr & ~TOP1[ge2]],
                 TOP5[anyr])
    t2 = TOP2[ge2]
    v = np.where(POP[ge2] >= 2, C2 | (t2 << 13) | TOP1[anyr & ~t2], v)
    has3 = ge3 != 0
    tb = TOP1[ge3]
    h3 = HI[ge3] << 13
    v = np.where(has3, C3 | h3 | TOP2[anyr & ~tb], v)
    st = STRAIGHT[anyr]
    v = np.where(st > 0, C4 | st, v)
    p2 = ge2 & ~tb
    v = np.where(has3 & (p2 != 0), C6 | h3 | HI[p2], v)
    v = np.where(quad != 0, C7 | (HI[quad] << 13) | HI[anyr & ~TOP1[quad]], v)
    fl = FLUSHV[a] | FLUSHV[b] | FLUSHV[c] | FLUSHV[d]
    v = np.maximum(v, fl)
    if not draws:
        return v
    fd = ((POP[a] == 4) | (POP[b] == 4) | (POP[c] == 4) | (POP[d] == 4)) & (v < C5)
    so = np.where(v < C4, np.minimum(SOUTS[anyr], 2), 0)
    return v, 0.85 * fd + 0.30 * so


_HD = np.array([0, 13, 8192, 13, 0, 0, 13, 13, 0], dtype=np.float64)
_LD = np.array([8192, 1, 1, 1, 14, 8192, 1, 1, 14], dtype=np.float64)


def value_to_float(v):
    """Hand value -> category + fraction in [0, 1): a smooth 'made hand' score."""
    cat = v >> 26
    hi = (v >> 13) & M13
    lo = v & M13
    hd = _HD[cat]
    sub = np.where(hd > 0, (hi + lo / 8192.0) / np.maximum(hd, 1.0), lo / _LD[cat])
    return cat + sub


def keep_score(key):
    """Heuristic value of a (2 hole + board) holding: made hand + draw potential."""
    v, bonus = eval_keys(key, draws=True)
    return value_to_float(v) + bonus


def keep_choice(k0, k1, k2, kflop):
    """Model of a sensible discard: keep the 2-card pair with the best keep_score."""
    best = k0 + k1
    bs = keep_score(best + kflop)
    for p in (k0 + k2, k1 + k2):
        s = keep_score(p + kflop)
        m = s > bs
        best = np.where(m, p, best)
        bs = np.where(m, s, bs)
    return best, bs


# ---- preflop heuristic score (Chen-like, on the three 2-card sub-hands) ----
_CHEN = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10], dtype=np.float64)
_GAPPEN = np.array([0, 0, 1, 2, 4, 5, 5], dtype=np.float64)


def _pair_score(c1, c2):
    r1 = c1 >> 2
    r2 = c2 >> 2
    hi = np.maximum(r1, r2)
    lo = np.minimum(r1, r2)
    isp = hi == lo
    sc = np.where(isp, np.maximum(_CHEN[hi] * 2.0, 5.0), _CHEN[hi])
    sc = sc + 2.0 * ((c1 & 3) == (c2 & 3))
    gap = np.clip(hi - lo, 0, 6)
    sc = sc - np.where(isp, 0.0, _GAPPEN[gap])
    sc = sc + ((gap >= 1) & (gap <= 2) & (hi < 10)) * 1.0
    return sc


def pre_score3(cards):
    """cards: (n, 3) int array -> heuristic preflop strength of each 3-card hand."""
    cards = np.asarray(cards, dtype=np.int64)
    s = np.stack([_pair_score(cards[:, 0], cards[:, 1]),
                  _pair_score(cards[:, 0], cards[:, 2]),
                  _pair_score(cards[:, 1], cards[:, 2])], axis=1)
    s.sort(axis=1)
    return s[:, 2] + 0.35 * s[:, 1] + 0.15 * s[:, 0]


# ---- exact preflop equity table (filled from PRE_EQ_DATA, see bottom) -----
def canon_index(cards):
    """Suit-isomorphism class index of a 3-card hand."""
    cs = sorted(cards, key=lambda x: (-(x >> 2), x & 3))
    r = [x >> 2 for x in cs]
    s = [x & 3 for x in cs]
    if r[0] != r[1] and r[1] != r[2]:
        if s[0] == s[1] == s[2]:
            p = 0
        elif s[0] == s[1]:
            p = 1
        elif s[0] == s[2]:
            p = 2
        elif s[1] == s[2]:
            p = 3
        else:
            p = 4
    elif r[0] == r[1] == r[2]:
        p = 0
    elif r[0] == r[1]:
        p = 1 if s[2] in (s[0], s[1]) else 0
    else:
        p = 1 if s[0] in (s[1], s[2]) else 0
    return ((r[0] * 13 + r[1]) * 13 + r[2]) * 5 + p


def canon_classes():
    """All (index, multiplicity) pairs in ascending index order (1755 classes)."""
    out = []
    for r0 in range(13):
        for r1 in range(r0 + 1):
            for r2 in range(r1 + 1):
                base = ((r0 * 13 + r1) * 13 + r2) * 5
                if r0 > r1 > r2:
                    out += [(base, 4), (base + 1, 12), (base + 2, 12), (base + 3, 12), (base + 4, 24)]
                elif r0 == r1 == r2:
                    out.append((base, 4))
                else:
                    out += [(base, 12), (base + 1, 12)]
    return out


# ---------------------------------------------------------------------------
# 2. Equity engine
# ---------------------------------------------------------------------------
KEEP_PAIRS = ((0, 1), (0, 2), (1, 2))      # option j keeps these hole indices ...
DISCARD_OF = (2, 1, 0)                      # ... and discards this one


class Spot:
    """
    A set of opponent holdings (sampled or enumerated) with, for each one, my
    equity against it and its current strength percentile.  Range assumptions
    are applied afterwards as weights, so one Spot serves a whole street.
      eq  : (n,) or (n, 3) -- 3 columns = my three keep options on the flop
      pct : (n,) weighted strength percentile of the opponent holding (1 = best)
      w0  : (n,) prior weight (card removal / discard consistency)
    """
    __slots__ = ("eq", "pct", "w0", "desc", "n")

    def __init__(self, eq, strength, w0=None):
        n = strength.shape[0]
        self.n = n
        self.eq = eq
        self.w0 = np.ones(n) if w0 is None else w0
        order = np.argsort(strength, kind="stable")
        ws = self.w0[order]
        cw = np.cumsum(ws)
        pct = np.empty(n)
        pct[order] = (cw - 0.5 * ws) / max(cw[-1], 1e-12)
        self.pct = pct
        self.desc = order[::-1]

    def weights(self, k, b):
        """Range density: mixture of a strength-biased part p^k and a flat part b."""
        if k <= 1e-6:
            return self.w0
        return self.w0 * ((1.0 - b) * (k + 1.0) * self.pct ** k + b)

    def equity(self, w, col=None):
        e = self.eq if col is None else self.eq[:, col]
        return w @ e / max(float(w.sum()), 1e-12)

    def vs_top(self, w, e, frac):
        """My equity against the strongest `frac` of the weighted range."""
        o = self.desc
        wo = w[o]
        cw = np.cumsum(wo)
        cut = int(np.searchsorted(cw, frac * cw[-1]))
        cut = min(max(cut, 0), self.n - 1)
        return float(wo[:cut + 1] @ e[o[:cut + 1]]) / max(float(cw[cut]), 1e-12)


def _remaining(used):
    u = set(used)
    return np.array([c for c in range(52) if c not in u], dtype=np.int64)


def _draw(rng, deck, n, k):
    """n independent draws of k distinct cards from deck -> (n, k) array."""
    idx = np.argpartition(rng.random((n, deck.shape[0])), k, axis=1)[:, :k]
    return deck[idx]


def sim_preflop(hole3, n, rng):
    """Preflop rollout: both sides see a flop, discard sensibly, then showdown."""
    deck = _remaining(hole3)
    dr = _draw(rng, deck, n, 8)                 # opp 0-2, flop 3-5, turn 6, river 7
    K = CARD_KEY[dr]
    kflop = K[:, 3] + K[:, 4] + K[:, 5]
    kboard = kflop + K[:, 6] + K[:, 7]
    h = [np.int64(CARD_KEY[c]) for c in hole3]
    mine, _ = keep_choice(h[0], h[1], h[2], kflop)
    opp, _ = keep_choice(K[:, 0], K[:, 1], K[:, 2], kflop)
    vm = eval_keys(mine + kboard)
    vo = eval_keys(opp + kboard)
    res = (vm > vo) + 0.5 * (vm == vo)
    return Spot(res, pre_strength_vec(dr[:, :3]))


def sim_flop(hole3, flop3, n, rng):
    """Flop rollout before the discard: one equity column per keep option."""
    deck = _remaining(list(hole3) + list(flop3))
    dr = _draw(rng, deck, n, 5)                 # opp 0-2, turn 3, river 4
    K = CARD_KEY[dr]
    kflop = np.int64(keys_of(flop3))
    kboard = kflop + K[:, 3] + K[:, 4]
    opp, oscore = keep_choice(K[:, 0], K[:, 1], K[:, 2], kflop)
    vo = eval_keys(opp + kboard)
    h = [np.int64(CARD_KEY[c]) for c in hole3]
    res = np.empty((n, 3))
    for j, (a, b) in enumerate(KEEP_PAIRS):
        vm = eval_keys(h[a] + h[b] + kboard)
        res[:, j] = (vm > vo) + 0.5 * (vm == vo)
    return Spot(res, oscore)


def _discard_consistency(c1, c2, opp_discard, flop3):
    """Down-weight opponent holdings that would not have discarded what we saw."""
    if opp_discard is None or flop3 is None or len(flop3) != 3:
        return None
    kf = np.int64(keys_of(flop3))
    kd = np.int64(CARD_KEY[opp_discard])
    k1 = CARD_KEY[c1]
    k2 = CARD_KEY[c2]
    kept = keep_score(k1 + k2 + kf)
    alt = np.maximum(keep_score(k1 + kd + kf), keep_score(k2 + kd + kf))
    return np.where(kept >= alt - 0.05, 1.0, 0.25)


def _combos(deck, rng, max_combos):
    i, j = np.triu_indices(deck.shape[0], 1)
    if max_combos and i.shape[0] > max_combos:
        sel = rng.choice(i.shape[0], max_combos, replace=False)
        i, j = i[sel], j[sel]
    return deck[i], deck[j]


def enum_turn(hole2, board4, dead, opp_discard, flop3, rng, max_combos=0):
    """Turn: every opponent 2-card holding x every river card (exact)."""
    deck = _remaining(list(hole2) + list(board4) + list(dead))
    c1, c2 = _combos(deck, rng, max_combos)
    kc = CARD_KEY[c1] + CARD_KEY[c2]
    kb = np.int64(keys_of(board4))
    kr = CARD_KEY[deck] + kb                                    # (m,) board incl. river
    m, nc = deck.shape[0], c1.shape[0]
    vo = eval_keys((kr[:, None] + kc[None, :]).ravel()).reshape(m, nc)
    vm = eval_keys(kr + np.int64(keys_of(hole2)))[:, None]
    valid = (deck[:, None] != c1[None, :]) & (deck[:, None] != c2[None, :])
    res = ((vm > vo) + 0.5 * (vm == vo)) * valid
    eq = res.sum(0) / np.maximum(valid.sum(0), 1)
    v4, bonus = eval_keys(kc + kb, draws=True)
    strength = value_to_float(v4) + 0.5 * bonus
    return Spot(eq, strength, _discard_consistency(c1, c2, opp_discard, flop3))


def enum_river(hole2, board5, dead, opp_discard, flop3, rng, max_combos=0):
    """River: exact equity against every opponent 2-card holding."""
    deck = _remaining(list(hole2) + list(board5) + list(dead))
    c1, c2 = _combos(deck, rng, max_combos)
    kb = np.int64(keys_of(board5))
    vo = eval_keys(CARD_KEY[c1] + CARD_KEY[c2] + kb)
    vm = eval_keys(np.array([keys_of(hole2)], dtype=np.int64) + kb)[0]
    eq = (vm > vo) + 0.5 * (vm == vo)
    return Spot(eq, vo.astype(np.float64), _discard_consistency(c1, c2, opp_discard, flop3))


def hand_percentile(hole2, board5):
    """Fraction of random 2-card holdings that a hand beats on a final board."""
    deck = _remaining(list(hole2) + list(board5))
    i, j = np.triu_indices(deck.shape[0], 1)
    kb = np.int64(keys_of(board5))
    vo = eval_keys(CARD_KEY[deck[i]] + CARD_KEY[deck[j]] + kb)
    vm = eval_keys(np.array([keys_of(hole2)], dtype=np.int64) + kb)[0]
    return float(np.mean((vm > vo) + 0.5 * (vm == vo)))


def draw_features(hole2, board):
    """(flush_draw, straight_outs_ranks, overcards) for my holding on a board."""
    key = keys_of(list(hole2) + list(board))
    v, _ = eval_keys(np.array([key], dtype=np.int64), draws=True)
    v = int(v[0])
    masks = [(key >> (13 * s)) & M13 for s in range(4)]
    anyr = masks[0] | masks[1] | masks[2] | masks[3]
    fd = 0
    if v < int(C5):
        for s in range(4):
            if int(POP[masks[s]]) == 4 and any((c & 3) == s for c in hole2):
                fd = 1
    so = int(SOUTS[anyr]) if v < int(C4) else 0
    top = max((c >> 2) for c in board) if board else -1
    over = sum(1 for c in hole2 if (c >> 2) > top)
    return fd, so, over


def board_wetness(board):
    """0 (dry) .. 1 (very draw-heavy) texture score."""
    if not board:
        return 0.0
    suits = [0, 0, 0, 0]
    mask = 0
    for c in board:
        suits[c & 3] += 1
        mask |= 1 << (c >> 2)
    w = 0.0
    mx = max(suits)
    if mx >= 3:
        w += 0.5
    elif mx == 2 and len(board) < 5:
        w += 0.3
    if int(STRAIGHT[mask]) or int(SOUTS[mask]) >= 1:
        w += 0.45
    else:
        rs = sorted(set(c >> 2 for c in board))
        close = sum(1 for x, y in zip(rs, rs[1:]) if y - x <= 2)
        w += 0.15 * min(close, 2)
    return min(w, 1.0)


# ---------------------------------------------------------------------------
# 3. Opponent model
# ---------------------------------------------------------------------------
ADAPT_AFTER = 20        # hands of baseline play before exploiting the stats
AGGR = ("bet", "raise")
AGG_STR_PRIOR = 0.72    # typical showdown percentile of a hand that bet postflop


def _k_of_mean(m):
    """Exponent k of a p^k range density whose mean strength percentile is m."""
    m = min(max(m, 0.5), 0.95)
    return (2.0 * m - 1.0) / (1.0 - m)


def c01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _sm(x, n, prior, k):
    """Beta-smoothed frequency."""
    return (x + prior * k) / (n + k)


class Profile:
    """Smoothed frequencies plus 0..1 archetype scores."""
    __slots__ = ("vpip", "pfr", "agg", "af", "ftb", "fts", "f3b", "pf3", "wtsd",
                 "bluff", "river_agg", "sd_win", "sd_str", "agg_str", "cal_n", "cal_d",
                 "station", "maniac", "nit", "bluffer", "conf")


MK_STATE = {"check": 1, "bet": 2, "call": 3, "raise": 4, "fold": 5}      # 0 = start of a flop
MK_PRIOR = np.tile(np.array([0.30, 0.50, 0.30, 0.50, 0.15, 0.45]), (6, 1))  # smoothing counts per row


class OppModel:
    """
    Leg-level statistics, built from canonical events
        (street 0..3, actor 'me'|'opp', kind fold|check|call|bet|raise, size)
    """

    def __init__(self):
        self.hands = 0
        self.vpip = self.pfr = 0
        self.agg = self.calls = self.checks = self.folds = 0     # postflop actions
        self.fb_n = self.fb_fold = self.fb_raise = 0             # facing my postflop bets
        self.steal_n = self.steal_fold = 0                       # my button open
        self.pf3_n = self.pf3 = 0                                # their 3-bets of my opens
        self.f3_n = self.f3_fold = 0                             # their fold to my re-raise
        self.saw_flop = self.showdowns = 0
        self.sd_wins = 0.0
        self.sd_str_sum = 0.0
        self.sd_str_n = 0
        self.agg_sd_sum = 0.0                                    # showdown strength after betting
        self.agg_sd_n = 0
        self.bluffs = self.values = 0                            # late aggression shown down
        self.river_seen = self.river_agg = 0
        self.cal_n = 0.0                                         # my turn / river calls shown down
        self.cal_d = 0.0                                         # sum of (estimated equity - result)
        self._cache = None
        self.tm = np.zeros((6, 6))                               # opponent action transition counts
        self._mk = None
        self.nit_scale = 1.0                                     # test switches for the nit score
        self.nit_aggcut = 0.0

    # ---- Markov model of the opponent's postflop actions -----------------------
    def markov(self):
        """
        (P, pi, lam2): row-stochastic transition matrix between the opponent's
        consecutive postflop actions (states MK_STATE, 0 = start of a flop),
        its stationary distribution pi (eigenvector of P' for eigenvalue 1) and
        the modulus of the second eigenvalue.  lam2 near 0 = the next action
        does not depend on the previous one; near 1 = strongly sequence-driven.
        """
        if self._mk is None:
            P = self.tm + MK_PRIOR
            P = P / P.sum(1, keepdims=True)
            try:
                w, v = np.linalg.eig(P.T)
                o = np.argsort(-np.abs(w))
                pi = np.abs(np.real(v[:, o[0]]))
                pi = pi / pi.sum()
                lam2 = float(np.abs(w[o[1]]))
            except Exception:
                pi = np.full(6, 1.0 / 6.0)
                lam2 = 0.0
            self._mk = (P, pi, lam2)
        return self._mk

    def markov_fold(self, prev):
        """
        P(opponent folds to a bet | their previous postflop action `prev`).
        The stationary (eigenvector) fold share is the baseline; the matrix row
        for `prev` replaces it in proportion to lam2 and to the row's sample.
        """
        P, pi, lam2 = self.markov()
        f_pi = pi[5] / max(pi[3] + pi[4] + pi[5], 1e-9)
        f_row = P[prev, 5] / max(P[prev, 3] + P[prev, 4] + P[prev, 5], 1e-9)
        row = self.tm[prev]
        n = row[3] + row[4] + row[5]
        w = min(1.0, lam2 / 0.5) * n / (n + 6.0)
        return float((1.0 - w) * f_pi + w * f_row)

    def note_call(self, est, result):
        """A call of mine reached showdown: remember how the estimate compared with the result."""
        self.cal_n += 1.0
        self.cal_d += est - result
        self._cache = None

    def update(self, ev, me_button, reached, showdown, opp_strength, opp_won):
        self.hands += 1
        self._cache = None
        self._mk = None
        if reached >= 1:                              # one pass through the action chain per flop seen
            prev = 0
            for e in ev:
                if e[1] == "opp" and e[0] >= 1 and e[2] in MK_STATE:
                    j = MK_STATE[e[2]]
                    self.tm[prev, j] += 1.0
                    prev = j
            self.tm[prev, 0] += 1.0                   # hand over: back to the start state
        pre = [e for e in ev if e[0] == 0]
        if any(e[1] == "opp" and e[2] in ("call", "bet", "raise") for e in pre):
            self.vpip += 1
        if any(e[1] == "opp" and e[2] in AGGR for e in pre):
            self.pfr += 1
        nr = 0
        for i, e in enumerate(pre):
            if e[2] not in AGGR:
                continue
            nr += 1
            if e[1] != "me":
                continue
            resp = next((x for x in pre[i + 1:] if x[1] == "opp"), None)
            if resp is None:
                continue
            if nr == 1 and me_button:
                self.steal_n += 1
                self.steal_fold += resp[2] == "fold"
                self.pf3_n += 1
                self.pf3 += resp[2] in AGGR
            elif nr >= 2:
                self.f3_n += 1
                self.f3_fold += resp[2] == "fold"
        for st in (1, 2, 3):
            se = [e for e in ev if e[0] == st]
            for i, e in enumerate(se):
                if e[1] == "opp":
                    if e[2] in AGGR:
                        self.agg += 1
                        if st == 3:
                            self.river_agg += 1
                    elif e[2] == "call":
                        self.calls += 1
                    elif e[2] == "check":
                        self.checks += 1
                    elif e[2] == "fold":
                        self.folds += 1
                elif e[2] in AGGR:
                    resp = next((x for x in se[i + 1:] if x[1] == "opp"), None)
                    if resp is not None:
                        self.fb_n += 1
                        self.fb_fold += resp[2] == "fold"
                        self.fb_raise += resp[2] in AGGR
        if reached >= 1:
            self.saw_flop += 1
        if reached >= 3:
            self.river_seen += 1
        if showdown:
            self.showdowns += 1
            if opp_won is not None:
                self.sd_wins += opp_won
            if opp_strength is not None:
                self.sd_str_sum += opp_strength
                self.sd_str_n += 1
                if any(e[1] == "opp" and e[0] >= 1 and e[2] in AGGR for e in ev):
                    self.agg_sd_sum += opp_strength          # what their bets turn out to be
                    self.agg_sd_n += 1
                if any(e[1] == "opp" and e[0] >= 2 and e[2] in AGGR for e in ev):
                    if opp_strength < 0.45:
                        self.bluffs += 1
                    else:
                        self.values += 1

    def profile(self):
        if self._cache is not None:
            return self._cache
        p = Profile()
        live = self.hands >= ADAPT_AFTER
        z = 1 if live else 0                 # baseline phase: priors only
        n = self.hands * z
        acts = (self.agg + self.calls + self.checks + self.folds) * z
        p.vpip = _sm(self.vpip * z, n, 0.75, 6)
        p.pfr = _sm(self.pfr * z, n, 0.45, 6)
        p.agg = _sm(self.agg * z, acts, 0.33, 10)
        p.af = (self.agg * z + 1.0) / (self.calls * z + 1.0)
        p.ftb = _sm(self.fb_fold * z, self.fb_n * z, 0.42, 8)
        p.fts = _sm(self.steal_fold * z, self.steal_n * z, 0.35, 6)
        p.f3b = _sm(self.f3_fold * z, self.f3_n * z, 0.50, 4)
        p.pf3 = _sm(self.pf3 * z, self.pf3_n * z, 0.12, 6)
        p.wtsd = _sm(self.showdowns * z, self.saw_flop * z, 0.45, 6)
        p.bluff = _sm(self.bluffs * z, (self.bluffs + self.values) * z, 0.30, 4)
        p.river_agg = _sm(self.river_agg * z, self.river_seen * z, 0.30, 6)
        p.sd_win = _sm(self.sd_wins * z, self.showdowns * z, 0.50, 6)
        p.sd_str = _sm(self.sd_str_sum * z, self.sd_str_n * z, 0.62, 4)
        p.agg_str = _sm(self.agg_sd_sum * z, self.agg_sd_n * z, AGG_STR_PRIOR, 5)
        p.cal_n, p.cal_d = self.cal_n * z, self.cal_d * z
        conf = c01(0.4 + (self.hands - ADAPT_AFTER) / 40.0) if live else 0.0
        p.conf = conf
        p.station = conf * c01((0.42 - p.ftb) / 0.20) * c01((p.vpip - 0.55) / 0.25) \
            * c01((0.50 - p.agg) / 0.20)
        p.maniac = conf * c01((p.agg - 0.42) / 0.22) * c01((p.pfr - 0.40) / 0.30 + 0.35)
        p.nit = conf * max(c01((p.ftb - 0.46) / 0.16),
                           0.5 * c01((0.55 - p.vpip) / 0.20) + 0.5 * c01((p.ftb - 0.40) / 0.15),
                           0.8 * c01((p.fts - 0.50) / 0.20))
        p.nit *= self.nit_scale
        if self.nit_aggcut > 0:                      # a player who bets this often is not a nit
            p.nit *= c01((self.nit_aggcut - p.agg) / 0.10)
        p.bluffer = conf * max(c01((p.bluff - 0.30) / 0.25),
                               c01((p.river_agg - 0.35) / 0.25) * c01((0.48 - p.sd_win) / 0.15))
        self._cache = p
        return p


_STREET_NAMES = {"preflop": 0, "pre-flop": 0, "pre_flop": 0, "pre": 0, "flop": 1,
                 "turn": 2, "river": 3, "0": 0, "1": 1, "2": 2, "3": 3}
_BTN_NAMES = ("button", "btn", "sb", "small_blind", "smallblind", "dealer")
_BB_NAMES = ("big_blind", "bb", "bigblind")
_ME_NAMES = ("me", "you", "hero", "self", "bot")
_OPP_NAMES = ("opp", "opponent", "villain", "other")


def parse_history(hist, me_button, sb=1.0, bb=2.0):
    """
    Convert the engine's action history
        {"street": "flop", "actor": "opp", "action": "raise", "amount": 6}
    into canonical events (street, 'me'|'opp', fold|check|call|bet|raise, size).
    The chips are replayed so that `size` is the bet or raise increment as a
    fraction of the pot after calling.  An opening bet arrives as "raise" and
    is relabelled "bet".  Returns None if the format is not understood, in
    which case the caller falls back on what it inferred live during the hand.
    """
    try:
        if isinstance(hist, dict):
            flat = []
            for name in ("preflop", "flop", "turn", "river"):
                for x in hist.get(name, []) or []:
                    if isinstance(x, dict):
                        x = dict(x)
                        x.setdefault("street", name)
                    flat.append(x)
            hist = flat
        if not isinstance(hist, list):
            return None
        out = []
        cur = None
        done = 0.0                      # chips in the pot from finished streets
        bets = {"me": 0.0, "opp": 0.0}  # chips committed on the street being replayed
        for x in hist:
            if not isinstance(x, dict):
                return None
            act = x.get("action", x.get("type", x.get("move")))
            if isinstance(act, dict):
                act = act.get("action", act.get("type"))
            act = str(act).lower()
            kind = next((k for k in ("fold", "check", "call", "raise", "bet") if act.startswith(k)), None)
            if kind is None:
                continue                                  # blinds, deals, discards ...
            st = x.get("street", x.get("round"))
            st = _STREET_NAMES.get(str(st).lower())
            if st is None:
                return None
            who = x.get("seat", x.get("player", x.get("actor", x.get("who", x.get("position")))))
            who = str(who).lower()
            if who in _ME_NAMES:
                actor = "me"
            elif who in _OPP_NAMES:
                actor = "opp"
            elif who in _BTN_NAMES:
                actor = "me" if me_button else "opp"
            elif who in _BB_NAMES:
                actor = "opp" if me_button else "me"
            else:
                return None
            if st != cur:
                if cur is not None:
                    done += bets["me"] + bets["opp"]
                if st == 0:
                    bets = {"me": sb if me_button else bb, "opp": bb if me_button else sb}
                else:
                    bets = {"me": 0.0, "opp": 0.0}
                cur = st
            other = "opp" if actor == "me" else "me"
            amt = x.get("amount")
            if isinstance(amt, bool) or not isinstance(amt, (int, float)):
                amt = None
            frac = 0.0
            if kind in AGGR:
                after_call = done + bets["me"] + bets["opp"] + max(0.0, bets[other] - bets[actor])
                new = float(amt) if amt is not None else bets[other] + 0.66 * after_call
                frac = max(0.0, new - bets[other]) / max(1.0, after_call)
                kind = "raise" if (st == 0 or max(bets["me"], bets["opp"]) > 0) else "bet"
                bets[actor] = new
            elif kind == "call":
                bets[actor] = float(amt) if amt is not None else bets[other]
            out.append((st, actor, kind, frac))
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. Bot: protocol, state tracking, time bank
# ---------------------------------------------------------------------------
_LOG = {"left": 40000}      # the engine caps stderr at 64 KB per leg: stay well under it


def log(*a):
    try:
        if _LOG["left"] <= 0:
            return
        s = " ".join(str(x) for x in a)[:400]
        _LOG["left"] -= len(s) + 2
        print(s, file=sys.stderr, flush=True)
    except Exception:
        pass


def num(d, keys, default=None):
    """First numeric value found under any of `keys` in dict d."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return default


_ACTS = ("fold", "check", "call", "raise", "bet")
_MIN_KEYS = ("min_raise", "min_raise_to", "min_to", "min_amount", "min_bet", "min")
_MAX_KEYS = ("max_raise", "max_raise_to", "max_to", "max_amount", "max_bet", "max")


def parse_legal(msg):
    """-> (set of legal action names, min_raise, max_raise, call amount or None)."""
    names = set()
    box = {"mn": None, "mx": None, "call": None}

    def rng(d):
        if box["mn"] is None:
            box["mn"] = num(d, _MIN_KEYS)
        if box["mx"] is None:
            box["mx"] = num(d, _MAX_KEYS)

    def scan(o, depth=0):
        if depth > 3:
            return
        if isinstance(o, str):
            if o.lower() in _ACTS:
                names.add(o.lower())
        elif isinstance(o, (list, tuple)):
            for x in o:
                scan(x, depth + 1)
        elif isinstance(o, dict):
            nm = o.get("action", o.get("type", o.get("name")))
            if isinstance(nm, str) and nm.lower() in _ACTS:
                nm = nm.lower()
                names.add(nm)
                if nm in AGGR:
                    rng(o)
                elif nm == "call" and box["call"] is None:
                    box["call"] = num(o, ("amount", "to_call", "call_amount"))
            else:
                rng(o)
            for k, v in o.items():
                kl = str(k).lower()
                if kl in _ACTS:
                    if v is False or v is None:
                        continue
                    names.add(kl)
                    if kl in AGGR:
                        if isinstance(v, dict):
                            rng(v)
                        elif isinstance(v, (list, tuple)) and len(v) == 2 \
                                and all(isinstance(t, (int, float)) for t in v):
                            box["mn"], box["mx"] = float(v[0]), float(v[1])
                    elif kl == "call" and isinstance(v, (int, float)) and not isinstance(v, bool):
                        box["call"] = float(v)
                elif isinstance(v, (list, dict)) and kl in ("actions", "legal", "options"):
                    scan(v, depth + 1)

    scan(msg.get("legal_actions", msg.get("legal", msg.get("actions"))))
    if box["mn"] is None:
        box["mn"] = num(msg, _MIN_KEYS[:5])
    if box["mx"] is None:
        box["mx"] = num(msg, _MAX_KEYS[:5])
    return names, box["mn"], box["mx"], box["call"]


class Hand:
    """Everything known about the hand in progress."""

    def __init__(self, button, hole, raws, sb, bb, stack):
        self.button = button            # True = I am the button (small blind, in position)
        self.hole = hole                # card ints (3, then 2 after the discard)
        self.raws = raws                # same cards exactly as the engine wrote them
        self.dealt = list(raws)         # the hand as first dealt, never shortened
        self.board = []
        self.flop = None
        self.street = 0
        self.commit_st = sb if button else bb   # my chips in on the current street
        self.total = self.commit_st             # my chips in this hand
        self.stack0 = stack
        self.raises_st = 0
        self.req_st = 0
        self.ev = []                    # canonical events, inferred live
        self.last_my = None             # (street, kind) of my latest action
        self.pf_agg = None              # last preflop raiser: 'me' | 'opp'
        self.my_discard = None
        self.opp_discard = None
        self.spots = {}                 # per-street equity cache
        self.trap = False               # checked a monster hoping to be bet into
        self.acted = False
        self.folded = False
        self.pre_spot = None
        self.ip = button                # the button acts last on flop, turn and river
        self.last_face = None           # (street, estimated equity, action) of my latest reply to a bet


class Bot:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.opp = OppModel()
        self.h = None
        self.sb, self.bb, self.stack = 1.0, 2.0, 200.0
        self.hands_total = 200
        self.hand_no = 0
        self.bank_ms = 30000.0
        self.time_left = None           # ms, as last reported by the engine
        self.used_ms = 0.0              # my own measured thinking time this leg
        self.hand_ms = 0.0
        self.cost1 = 16.0               # estimated ms per hand at scale 1.0
        self.comp_ema = 0.0
        self.drain_ema = None
        self.tl_hand_start = None
        self.scale_now = 1.0
        self.net = 0.0

    # ---- time bank --------------------------------------------------------
    def note_time(self, m):
        t = num(m, ("time_left_ms", "time_bank_ms", "timebank_ms", "time_remaining_ms"))
        if t is None:
            t = num(m, ("time_left", "time_bank", "time_remaining"))
            if t is not None and t <= 600:          # looks like seconds
                t *= 1000.0
        if t is not None:
            self.time_left = t

    def _time_left(self):
        return self.time_left if self.time_left is not None else self.bank_ms - self.used_ms

    def compute_scale(self):
        """Rollout scale so projected spend stays well inside the remaining bank."""
        tl = self._time_left()
        left = max(1, self.hands_total - self.hand_no + 1)
        cushion = max(1200.0, 0.12 * self.bank_ms)
        per = 0.6 * (tl - cushion) / left
        overhead = 0.0
        if self.drain_ema is not None:
            overhead = max(0.0, self.drain_ema - self.comp_ema)   # time I am billed but do not use
        s = (per - overhead) / max(self.cost1, 1.0)
        self.scale_now = max(0.0, min(2.5, s))
        return self.scale_now

    def panic(self):
        # An exhausted time bank forfeits the whole match: stop thinking early.
        return self._time_left() < 600.0

    # ---- messages ---------------------------------------------------------
    def on_init(self, m):
        self.opp = OppModel()
        self.h = None
        self.hand_no = 0
        self.net = 0.0
        blinds = m.get("blinds")
        if isinstance(blinds, (list, tuple)) and len(blinds) >= 2:
            self.sb, self.bb = float(blinds[0]), float(blinds[1])
        elif isinstance(blinds, dict):
            self.sb = num(blinds, ("small", "sb", "small_blind"), self.sb)
            self.bb = num(blinds, ("big", "bb", "big_blind"), self.bb)
        self.sb = num(m, ("small_blind", "sb"), self.sb)
        self.bb = num(m, ("big_blind", "bb"), self.bb)
        if self.bb <= 0 or self.sb <= 0:
            self.sb, self.bb = 1.0, 2.0
        self.stack = num(m, ("starting_stack", "stack_size", "stack", "stacks", "chips"), 200.0)
        if self.stack < 2 * self.bb:
            self.stack = 200.0
        self.hands_total = int(num(m, ("hands", "num_hands", "hands_per_leg", "n_hands",
                                       "total_hands", "leg_hands"), 200))
        if self.hands_total <= 0:
            self.hands_total = 200
        self.time_left = None
        self.note_time(m)
        if self.time_left is not None:
            self.bank_ms = self.time_left
        else:
            self.bank_ms = 30000.0 if self.hands_total <= 200 else 75000.0
        self.used_ms = 0.0
        self.comp_ema = 0.0
        self.drain_ema = None
        self.tl_hand_start = None

    def on_new_hand(self, m):
        self.note_time(m)
        hid = num(m, ("hand_id",))
        self.hand_no = int(hid) if hid is not None and hid >= 1 else self.hand_no + 1
        if self.hand_no > self.hands_total:         # leg is longer than announced
            self.hands_total = 500 if self.hand_no <= 500 else self.hand_no + 200
        seat = str(m.get("seat", m.get("position", ""))).lower()
        button = seat in _BTN_NAMES
        hole, raws = parse_cards(m.get("hole_cards", m.get("cards", m.get("hand"))))
        self.h = Hand(button, hole, raws, self.sb, self.bb,
                      num(m, ("stack",), self.stack))
        dealt = m.get("hole_cards", m.get("cards", m.get("hand")))
        if isinstance(dealt, (list, tuple)) and dealt:
            self.h.dealt = list(dealt)              # kept verbatim even if unparseable
        self.hand_ms = 0.0
        self.tl_hand_start = self.time_left
        self.compute_scale()

    def on_discard_result(self, m):
        h = self.h
        if h is None:
            return
        self.note_time(m)
        hole, raws = parse_cards(m.get("hole_cards", m.get("cards")))
        if len(hole) == 2:
            for c in h.hole:
                if c not in hole:
                    h.my_discard = c
            h.hole, h.raws = hole, raws
        elif len(h.hole) == 3 and h.my_discard in h.hole:
            i = h.hole.index(h.my_discard)
            h.hole = h.hole[:i] + h.hole[i + 1:]
            h.raws = h.raws[:i] + h.raws[i + 1:]
        od = m.get("opp_discard", m.get("opponent_discard"))
        if isinstance(od, (list, tuple)) and od:
            od = od[0]
        h.opp_discard = parse_card(od) if od is not None else None
        md = parse_card(m.get("my_discard", m.get("discard"))) if \
            (m.get("my_discard", m.get("discard")) is not None) else None
        if md is not None:
            h.my_discard = md

    def on_other(self, m):
        """Unknown message type: only pick up a board / clock if it carries one."""
        self.note_time(m)
        if str(m.get("type", "")).lower() == "notice":      # the engine corrected a reply of mine
            log("NOTICE hand %d: %s -> %s" % (self.hand_no, m.get("code"), m.get("applied")))
            return
        h = self.h
        if h is None:
            return
        board, _ = parse_cards(m.get("board", m.get("community_cards")))
        if len(board) > len(h.board) and len(board) <= 5:
            h.board = board
            if h.flop is None and len(board) >= 3:
                h.flop = board[:3]

    # ---- live inference of the opponent's actions --------------------------
    def advance(self, street, to_call, pot, infer=True):
        """
        Enter the street.  With infer=True also deduce what the opponent just
        did from the change in state; that is only needed when a request
        carries no usable history (the engine normally sends the full record).
        """
        h = self.h
        if street != h.street:
            lm = h.last_my
            if infer and lm is not None and lm[0] == h.street:
                if lm[1] in AGGR:
                    h.ev.append((h.street, "opp", "call", 0.0))
                elif lm[1] == "check" and h.street >= 1 and not h.ip:
                    h.ev.append((h.street, "opp", "check", 0.0))
            h.street = street
            h.commit_st = 0.0
            h.raises_st = 0
            h.req_st = 0
        first = h.req_st == 0
        h.req_st += 1
        if not infer:
            return
        frac = to_call / max(1.0, pot - to_call)
        lm = h.last_my if (h.last_my is not None and h.last_my[0] == street) else None
        if street == 0:
            if to_call > 0 and (not first or not h.button):
                h.ev.append((0, "opp", "raise", frac))
                h.raises_st += 1
                h.pf_agg = "opp"
            elif first and not h.button:
                h.ev.append((0, "opp", "call", 0.0))
        elif first:
            if h.ip:                                       # opponent acts first postflop
                if to_call > 0:
                    h.ev.append((street, "opp", "bet", frac))
                    h.raises_st += 1
                else:
                    h.ev.append((street, "opp", "check", 0.0))
        elif to_call > 0:
            kind = "bet" if (lm is not None and lm[1] == "check") else "raise"
            h.ev.append((street, "opp", kind, frac))
            h.raises_st += 1

    def record_my(self, kind, to_call, pot, amount=None):
        """Book-keep my own action; returns the canonical kind."""
        h = self.h
        frac = 0.0
        if kind == "call":
            h.commit_st += to_call
            h.total += to_call
        elif kind == "raise":
            frac = max(0.0, amount - h.commit_st - to_call) / max(1.0, pot + to_call)
            h.total += amount - h.commit_st
            h.commit_st = amount
            h.raises_st += 1
            if h.street == 0:
                h.pf_agg = "me"
            elif to_call <= 0:
                kind = "bet"
        elif kind == "fold":
            h.folded = True
        h.ev.append((h.street, "me", kind, frac))
        h.last_my = (h.street, kind)
        h.acted = True
        return kind

    # ---- end of hand ------------------------------------------------------
    def on_hand_result(self, m):
        h = self.h
        if h is None:
            return
        self.note_time(m)
        try:
            self._finish_hand(m, h)
        except Exception as e:                              # never let stats kill the bot
            log("hand_result error:", repr(e))
        # time accounting for the adaptive rollout scale
        self.comp_ema = self.hand_ms if self.comp_ema == 0 else 0.85 * self.comp_ema + 0.15 * self.hand_ms
        if self.scale_now > 0.15 and self.hand_ms > 0:
            c = self.hand_ms / self.scale_now
            self.cost1 = 0.85 * self.cost1 + 0.15 * min(c, 200.0)
        if self.tl_hand_start is not None and self.time_left is not None:
            d = self.tl_hand_start - self.time_left
            if 0 <= d < 5000:
                self.drain_ema = d if self.drain_ema is None else 0.85 * self.drain_ema + 0.15 * d
        self.h = None

    def _finish_hand(self, m, h):
        oc, _ = parse_cards(m.get("opp_hole_cards", m.get("opponent_hole_cards", m.get("opp_cards"))))
        showdown = len(oc) >= 2 or str(m.get("result", "")).lower() == "showdown"
        mine_now, mine_raw = parse_cards(m.get("my_hole_cards"))
        if len(mine_now) in (2, 3):
            h.hole, h.raws = mine_now, mine_raw
        od = m.get("opp_discard", m.get("opponent_discard"))
        if h.opp_discard is None and od is not None:
            h.opp_discard = parse_card(od[0] if isinstance(od, (list, tuple)) and od else od)
        board, _ = parse_cards(m.get("board", m.get("community_cards")))
        if len(board) < len(h.board):
            board = h.board
        # finish the live event list with what the result reveals
        lm = h.last_my
        if not h.folded:
            if lm is None:
                if not h.button and not showdown:
                    h.ev.append((0, "opp", "fold", 0.0))          # button gave up the blind
            elif lm[1] in AGGR:
                h.ev.append((lm[0], "opp", "call" if showdown else "fold", 0.0))
            elif lm[1] == "check" and showdown and lm[0] >= 1 and not h.ip:
                h.ev.append((lm[0], "opp", "check", 0.0))
        ev = h.ev
        parsed = parse_history(m.get("history", m.get("actions")), h.button, self.sb, self.bb)
        if parsed is not None and (parsed or not ev):
            ev = parsed                                 # complete record from the engine
        reached = max([h.street] + [e[0] for e in ev])
        if showdown:
            reached = max(reached, 3 if len(board) == 5 else reached)
        strength = None
        opp_won = None
        if showdown and len(board) == 5:
            kb = keys_of(board)
            best = None
            for i in range(len(oc)):
                for j in range(i + 1, len(oc)):
                    v = int(eval_keys(np.array([kb + keys_of([oc[i], oc[j]])], dtype=np.int64))[0])
                    if best is None or v > best[0]:
                        best = (v, [oc[i], oc[j]])
            if best is not None and not (set(best[1]) & set(board)):
                strength = hand_percentile(best[1], board)
                mine = [c for c in h.hole if c not in board]
                if len(mine) >= 2:
                    mv = max(int(eval_keys(np.array([kb + keys_of([mine[i], mine[j]])], dtype=np.int64))[0])
                             for i in range(len(mine)) for j in range(i + 1, len(mine)))
                    opp_won = 1.0 if best[0] > mv else (0.5 if best[0] == mv else 0.0)
        self.opp.update(ev, h.button, reached, showdown, strength, opp_won)
        r = m.get("result")
        d = None
        if isinstance(r, (int, float)) and not isinstance(r, bool):
            d = float(r)
        elif isinstance(r, dict):
            d = num(r, ("delta", "payoff", "chips", "net", "winnings", "amount", "you", "me"))
        if d is None:
            d = num(m, ("delta", "payoff", "net", "winnings", "chips_won"))
        if d is not None:
            self.net += d
        lf = h.last_face
        if showdown and d is not None and lf is not None and lf[2] == "call" and lf[0] >= 2:
            self.opp.note_call(lf[1], 1.0 if d > 0 else (0.5 if d == 0 else 0.0))
        if self.hand_no % 50 == 0:
            p = self.opp.profile()
            log("hand %d net %+.0f | vpip %.2f pfr %.2f agg %.2f ftb %.2f fts %.2f wtsd %.2f bluff %.2f "
                "rivagg %.2f sdwin %.2f | station %.2f maniac %.2f nit %.2f bluffer %.2f | "
                "scale %.2f t_left %.0f" %
                (self.hand_no, self.net, p.vpip, p.pfr, p.agg, p.ftb, p.fts, p.wtsd, p.bluff,
                 p.river_agg, p.sd_win, p.station, p.maniac, p.nit, p.bluffer,
                 self.scale_now, self._time_left()))


def sim_flop2(hole2, flop3, dead, opp_discard, n, rng):
    """Flop rollout when both players already hold two cards."""
    deck = _remaining(list(hole2) + list(flop3) + list(dead))
    dr = _draw(rng, deck, n, 4)                 # opp 0-1, turn 2, river 3
    K = CARD_KEY[dr]
    kf = np.int64(keys_of(flop3))
    kboard = kf + K[:, 2] + K[:, 3]
    ko = K[:, 0] + K[:, 1]
    vo = eval_keys(ko + kboard)
    vm = eval_keys(np.int64(keys_of(hole2)) + kboard)
    res = (vm > vo) + 0.5 * (vm == vo)
    return Spot(res, keep_score(ko + kf), _discard_consistency(dr[:, 0], dr[:, 1], opp_discard, flop3))


# ---- preflop strength lookups ---------------------------------------------
PRE_EQ = np.zeros(13 * 13 * 13 * 5)     # equity vs a random hand, by class index
PRE_PCT = np.zeros(13 * 13 * 13 * 5)    # percentile of that equity over all hands
_PRE = {"have": False, "sorted": None}


def canon_index_vec(cards):
    """Vectorised canon_index for an (n, 3) card array."""
    r = cards >> 2
    order = np.argsort(-r, axis=1, kind="stable")
    r = np.take_along_axis(r, order, 1)
    s = np.take_along_axis(cards & 3, order, 1)
    r0, r1, r2 = r[:, 0], r[:, 1], r[:, 2]
    e01 = s[:, 0] == s[:, 1]
    e02 = s[:, 0] == s[:, 2]
    e12 = s[:, 1] == s[:, 2]
    pd = np.where(e01 & e12, 0, np.where(e01, 1, np.where(e02, 2, np.where(e12, 3, 4))))
    trips = (r0 == r1) & (r1 == r2)
    p = np.where((r0 != r1) & (r1 != r2), pd,
                 np.where(trips, 0, np.where(r0 == r1, e02 | e12, e01 | e02)))
    return ((r0 * 13 + r1) * 13 + r2) * 5 + p


def pre_strength_vec(cards):
    cards = np.asarray(cards, dtype=np.int64)
    if _PRE["have"]:
        return PRE_EQ[canon_index_vec(cards)]
    return pre_score3(cards)


def pre_strength(hole3):
    """(equity vs random hand, percentile among all starting hands)."""
    if _PRE["have"]:
        i = canon_index(hole3)
        return float(PRE_EQ[i]), float(PRE_PCT[i])
    if _PRE["sorted"] is None:                      # fallback: heuristic score only
        import itertools
        allc = np.array(list(itertools.combinations(range(52), 3)), dtype=np.int64)
        _PRE["sorted"] = np.sort(pre_score3(allc))
    sc = float(pre_score3(np.array([hole3], dtype=np.int64))[0])
    pct = float(np.searchsorted(_PRE["sorted"], sc)) / _PRE["sorted"].shape[0]
    return 0.35 + 0.35 * pct, pct


def load_pre_table(data):
    cls = canon_classes()
    if len(data) != 3 * len(cls):
        return False
    idx = np.array([c[0] for c in cls], dtype=np.int64)
    mult = np.array([c[1] for c in cls], dtype=np.float64)
    vals = np.array([int(data[i:i + 3]) for i in range(0, len(data), 3)], dtype=np.float64) / 1000.0
    order = np.argsort(vals, kind="stable")
    ws = mult[order]
    cw = np.cumsum(ws)
    pct = np.empty(len(cls))
    pct[order] = (cw - 0.5 * ws) / cw[-1]
    PRE_EQ[idx] = vals
    PRE_PCT[idx] = pct
    _PRE["have"] = True
    return True


# ---------------------------------------------------------------------------
# 5. Strategy
# ---------------------------------------------------------------------------
class GameState:
    """One action request, as the decision engine sees it."""
    __slots__ = ("street", "to_call", "pot", "mn", "mx", "can_raise", "names", "stack_left",
                 "engine", "tier", "my_bet")


def clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class WhiteMagic:
    """
    'White Magic' layer: Hellmuth-style small-ball and disciplined-fold knobs
    on top of the equity / EV engine.  Every knob at its neutral value leaves
    the engine untouched, so each one is set by A/B testing, not by taste.

      survival_factor        0..1  extra equity demanded before risking most of
                                   the stack (waived when pot odds beat 2:1)
      pot_control_threshold  0..1  share of the stack a one-pair hand may commit
                                   on the turn / river of a single-raised pot
                                   (1 = no cap)
      hero_fold_sensitivity  0..1  equity discount when a passive player
                                   (aggression frequency < 25%) raises, or bets
                                   more than 65% pot, on the turn / river
      trap_frequency         0..1  chance to flat-call instead of re-raising
                                   with a top-3% hand against a loose-aggressive
                                   player (VPIP > 35%, aggression frequency > 45%)

    There is no ICM in this format (stacks reset to 200 every hand and nobody
    is eliminated), so survival_factor only buys lower variance, never equity.
    """

    def __init__(self, **kw):
        self.survival_factor = 0.0
        self.pot_control_threshold = 1.0
        self.hero_fold_sensitivity = 0.0
        self.trap_frequency = 0.0
        self.small_ball = 0.0           # 1: 30-40% pot bets on dry boards
        self.oop_tightness = 0.0        # 0..1: tighter big-blind defence versus opens
        self.limp_speculative = 0.0     # 1: limp small pairs / suited connectors on the button
        self.check_back_medium = 0.0    # chance to check back one pair in position on wet boards
        self.nit_strict_fold = 0.0      # 1: versus nits, fold below two pair to a raise / big bet
        self.station_sizes = (0.9, 1.5, 2.2)    # value sizes versus calling stations (A/B tested 14 Nov)
        self.maniac_patience = 0.0      # 0..1: wait for top-10% hands preflop versus maniacs
        self.lag_dry_call = 0.0         # 0..1: call wider on dry boards versus aggressive players
        self.hellmuth = 0.0             # 1: Hellmuth style pack (nit probes 30-40% and strict folds,
        #                                 no bluffs and 75-80% value bets versus stations, push/fold at M < 10)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError("unknown White Magic knob: %s" % k)
            setattr(self, k, v)


WM = WhiteMagic()       # the values the bot plays with
TRACE = False           # True: log every decision on stderr (local debugging only; stderr is capped)

# Strategy constants.  Every value here was set by duplicate-deal benchmarking
# (see BENCHMARK.md / STRATEGIES.md); Player(...).P can override them for A/B tests.
PARAMS = dict(
    # preflop: hand-percentile thresholds, sizes in big blinds, equity realisation
    open_thr=0.30, limp_thr=0.10, open_size=2.5, iso_thr=0.62, iso_size=4.0,
    real_ip=0.95, real_oop=0.78, real3_ip=0.85, real3_oop=0.78,
    smin_base=0.25, smin_slope=0.08, thr_3bet=0.86, thr_4bet=0.93,
    k_open=0.6, k_3bet=1.8,
    # opponent range model after the flop (see Player.range_params)
    k_bet=0.7, k_bet_size=0.9, k_raise=1.7, k_call=0.5, k_check=0.3, b_bet=0.22, b_raise=0.10,
    # betting when checked to: sizes as pot fractions, fold model, value and bluff rules
    sizes_dry=(1.0, 1.5), sizes_wet=(1.25, 2.0), sizes_river=(),
    fold_base=0.6, fold_slope=0.6, val_thr=0.50, val_gain=0.02, val_sure=0.56,
    bluff_draw=0.70, bluff_ip=0.70, bluff_oop=0.55, bluff_gain=0.0,
    # facing a bet: safety margin over pot odds, raise thresholds
    need_river=0.03, need_early=0.01, raise_thr=0.70, raise_top=0.52,
    # learning within a leg
    cal_gain=0.0, cal_n0=8.0, cal_cap=0.15, cal_cap_lo=0.05,     # showdown calibration of calls
    loose_3bet=0.0,          # how much wider to 3-bet players who open nearly every button
    agg_gamma=1.0,           # 1 = a frequent bettor's bets count for less; 0 = no such discount
    nit_scale=1.0, nit_aggcut=0.0,   # nit score multiplier / aggression above which nobody is a nit (0 = off)
    mk_gain=0.0,             # weight of the Markov (transition matrix / eigenvalue) fold estimate
)


def made_tier(hole, board):
    """0 = nothing of my own, 1 = a one-pair hand, 2 = two pair or better with my cards."""
    if len(board) < 3 or len(hole) < 2:
        return 0
    mine = int(eval_keys(np.array([keys_of(list(hole) + list(board))], dtype=np.int64))[0]) >> 26
    shared = int(eval_keys(np.array([keys_of(board)], dtype=np.int64))[0]) >> 26
    if mine == shared or mine == 0:
        return 0
    if mine >= 3:
        return 2
    if mine == 2:
        return 2 if shared == 0 else 1
    return 1


def is_speculative(hole):
    """Small pocket pair or suited connector somewhere in the hand (set-mining material)."""
    for i in range(len(hole)):
        for j in range(i + 1, len(hole)):
            r1, r2 = hole[i] >> 2, hole[j] >> 2
            if r1 == r2 and r1 <= 7:
                return True
            if (hole[i] & 3) == (hole[j] & 3) and abs(r1 - r2) == 1:
                return True
    return False


def evaluate_action(game_state, opponent_stats):
    """
    The decision entry point: returns (action, raise_to_amount).
      game_state      GameState built from the engine's action_request
      opponent_stats  Profile of the opponent (OppModel.profile())
    The chosen action is the best-scoring one under the equity / EV model,
    after the White Magic adjustments for this opponent have been applied.
    """
    return game_state.engine.evaluate_action(game_state, opponent_stats)


class Player(Bot):
    def __init__(self, wm=None):
        Bot.__init__(self)
        self.wm = wm if wm is not None else WM
        self.P = dict(PARAMS)
        self.dbg = {}

    def on_init(self, m):
        Bot.on_init(self, m)
        self.opp.nit_scale = self.P["nit_scale"]
        self.opp.nit_aggcut = self.P["nit_aggcut"]

    # ---- action request ---------------------------------------------------
    def on_action_request(self, m):
        self.note_time(m)
        names, mn, mx, call_amt = parse_legal(m)
        if self.h is None:                                   # defensive: no new_hand seen
            hole, raws = parse_cards(m.get("hole_cards"))
            seat = str(m.get("seat", "")).lower()
            self.h = Hand(seat in _BTN_NAMES, hole, raws, self.sb, self.bb, self.stack)
        h = self.h
        board, _ = parse_cards(m.get("board", m.get("community_cards", m.get("board_cards"))))
        if len(board) >= len(h.board):
            h.board = board
        if h.flop is None and len(h.board) >= 3:
            h.flop = h.board[:3]
        hc, hr = parse_cards(m.get("hole_cards"))
        if len(hc) in (2, 3) and len(hc) <= len(h.hole) and set(hc) != set(h.hole):
            if len(hc) == 2 and len(h.hole) == 3:
                h.my_discard = next((x for x in h.hole if x not in hc), h.my_discard)
            h.hole, h.raws = hc, hr
        street = None
        sn = m.get("street", m.get("round"))
        if sn is not None:
            street = _STREET_NAMES.get(str(sn).lower())
        if street is None:
            street = {0: 0, 3: 1, 4: 2, 5: 3}.get(len(h.board), h.street)
        street = max(street, h.street)
        if street >= 2 and len(h.hole) == 3 and h.my_discard in h.hole:
            i = h.hole.index(h.my_discard)
            h.hole = h.hole[:i] + h.hole[i + 1:]
            h.raws = h.raws[:i] + h.raws[i + 1:]

        to_call = num(m, ("to_call", "call_amount", "amount_to_call"))
        if to_call is None:
            if "check" in names:
                to_call = 0.0
            elif call_amt is not None:
                to_call = call_amt
            else:
                to_call = 0.0 if not names else self.bb
        to_call = max(0.0, to_call)
        pot = num(m, ("pot", "pot_size", "pot_total"))
        if pot is None or pot <= 0:
            pot = 2.0 * h.total + to_call                    # heads-up identity
        # The engine's own record of the hand is authoritative: use it for the
        # opponent's actions and for my chip counts whenever it is present.
        ev = None
        if isinstance(m.get("history"), list):
            ev = parse_history(m["history"], h.button, self.sb, self.bb)
        self.advance(street, to_call, pot, infer=ev is None)
        if ev is not None:
            h.ev = ev
            pre = [x for x in ev if x[0] == 0 and x[2] in AGGR]
            h.pf_agg = pre[-1][1] if pre else None
        mb = num(m, ("my_bet",))
        if mb is not None:
            h.commit_st = mb
        ms = num(m, ("my_stack",))
        if ms is not None:
            h.total = max(0.0, h.stack0 - ms)
        rl = num(m, ("raises_left",))
        if rl is not None:
            h.raises_st = 4 - int(rl)                        # preflop the big blind is the first

        if not names:
            names = {"fold", "call", "raise"} if to_call > 0 else {"check", "raise"}
        stack_left = ms if ms is not None else max(0.0, h.stack0 - h.total)
        can_raise = ("raise" in names) or ("bet" in names)
        if can_raise:
            if mx is None:
                mx = h.commit_st + stack_left
            if mn is None:
                mn = h.commit_st + to_call + max(to_call, self.bb)
            mn = min(mn, mx)
            if mx <= h.commit_st + to_call:
                can_raise = False
            elif ms is None:
                stack_left = mx - h.commit_st            # max_raise is the all-in "to" amount
        c = GameState()
        c.street, c.to_call, c.pot, c.mn, c.mx = street, to_call, pot, mn, mx
        c.can_raise, c.names, c.stack_left = can_raise, names, stack_left
        c.engine, c.tier, c.my_bet = self, 0, h.commit_st

        self.dbg = {}
        try:
            act, amt = evaluate_action(c, self.opp.profile())
        except Exception as e:
            log("decide error:", repr(e))
            act, amt = ("check", 0) if to_call <= 0 else \
                (("call", 0) if to_call <= 0.2 * pot else ("fold", 0))
        reply = self.finish_action(c, act, amt)
        if TRACE:
            try:
                self.trace(c, reply)
            except Exception:
                pass
        return reply

    def trace(self, c, reply):
        """Decision log on stderr: Analyze / Range Assessment / Action / Rationale."""
        h, d, p = self.h, self.dbg, self.opp.profile()
        kind, score = max((("calling station", p.station), ("maniac", p.maniac), ("nit", p.nit),
                           ("bluffer", p.bluffer)), key=lambda x: x[1])
        arche = kind if score >= 0.4 else ("unknown (baseline phase)" if p.conf <= 0 else "balanced")
        board = " ".join(RANKS[x >> 2] + SUITS[x & 3] for x in h.board)
        tex = "preflop" if not h.board else (("wet " if board_wetness(h.board) >= 0.5 else "dry ") + board)
        eq, need = d.get("eq"), d.get("need")
        if "k" in d:
            rng = "weighted to strength k=%.1f with %.0f%% air; my equity %.0f%%" % (d["k"], 100 * d["b"], 100 * eq)
        elif eq is not None:
            rng = "raising range; my equity %.0f%%, hand percentile %.0f%%" % (100 * eq, 100 * d.get("s", 0))
        else:
            rng = "no information yet; hand percentile %.0f%%" % (100 * d.get("s", 0))
        act = reply["action"]
        if act == "raise":
            action = ("RAISE TO %d" % reply["amount"]) if c.to_call > 0 else \
                ("BET %d%% (to %d)" % (round(100.0 * (reply["amount"] - c.my_bet) / max(c.pot, 1)),
                                       reply["amount"]))
            why = "builds the pot with the best of it or takes it down now"
        elif act == "call":
            action = "CALL"
            why = "equity beats the price" + ("" if need is None else " of %.0f%%" % (100 * need))
        elif act == "fold":
            action = "FOLD"
            why = "equity is below the price" + ("" if need is None else " of %.0f%%" % (100 * need)) + \
                "; chips preserved"
        else:
            action = "CHECK"
            why = "no bet beats checking; pot kept small"
        log("Analyze: [%.0f BB, hand %d/%d, %s, %s] | Range Assessment: [%s] | Action: %s | Rationale: %s"
            % (c.stack_left / self.bb, self.hand_no, self.hands_total, arche, tex, rng, action, why))

    def finish_action(self, c, act, amt):
        """Legalise the chosen action, book-keep it, and build the reply."""
        names = c.names
        if act == "raise" and not c.can_raise:
            act = "call" if c.to_call > 0 else "check"
        if act == "fold" and c.to_call <= 0:
            act = "check"
        if act == "call" and c.to_call <= 0 and "check" in names:
            act = "check"
        if act == "check" and (c.to_call > 0 or "check" not in names):
            if c.to_call <= 0 and "call" in names:
                act = "call"
            elif c.to_call > 0:
                act = "fold"
        if act == "call" and "call" not in names:
            act = "check" if "check" in names else ("fold" if "fold" in names else "call")
        if act == "fold" and "fold" not in names:
            act = "check" if "check" in names else "call"
        if act == "raise":
            lo, hi = int(np.ceil(c.mn - 1e-9)), int(np.floor(c.mx + 1e-9))
            if hi < lo:
                lo = hi
            a = int(round(amt))
            a = max(lo, min(hi, a))
            self.record_my("raise", c.to_call, c.pot, float(a))
            return {"type": "action", "action": "raise" if "raise" in names else "bet", "amount": a}
        self.record_my(act, c.to_call, c.pot)
        return {"type": "action", "action": act}

    def evaluate_action(self, c, p):
        """(action, raise_to) for GameState c against opponent Profile p."""
        if self.panic() or len(self.h.hole) < 2:
            return self.panic_action(c)
        if c.street == 0 and len(self.h.hole) == 3:
            return self.decide_preflop(c, p)
        return self.decide_postflop(c, p)

    def panic_action(self, c):
        """Out of time (or no cards known): constant-time, never bleeds chips."""
        if c.to_call <= 0:
            return "check", 0
        h = self.h
        if c.street == 0 and len(h.hole) == 3:
            _, s = pre_strength(h.hole)
            if s >= 0.80 or (s >= 0.35 and c.to_call <= self.bb):
                return "call", 0
        elif c.to_call <= 0.12 * c.pot:
            return "call", 0
        return "fold", 0

    # ---- opponent range model -----------------------------------------------
    def range_params(self, street, p):
        """
        (k, b) for Spot.weights: k sharpens the opponent's range towards strong
        holdings according to the actions they took, b is the share of the
        range that is unrelated to strength (bluffs, floats, noise).
        """
        h = self.h
        P = self.P
        k = 0.0          # from passive actions and preflop
        ka = 0.0         # from postflop bets and raises
        my_pf_raises = 0
        last = None
        call_w = clip(p.ftb / 0.42, 0.4, 1.5)        # a station's call says little
        for st, who, kind, frac in h.ev:
            if who == "me":
                if st == 0 and kind in AGGR:
                    my_pf_raises += 1
                continue
            if st == street:
                last = (kind, frac)
            if st == 0:
                if kind in AGGR:
                    if my_pf_raises == 0:
                        inc = P["k_open"] * clip(0.5 / p.pfr, 0.5, 2.5)
                    elif my_pf_raises == 1:
                        inc = P["k_3bet"] * clip(0.15 / p.pf3, 0.5, 2.5)
                    else:
                        inc = 2.2
                elif kind == "call":
                    inc = 0.2
                else:
                    inc = 0.0
                k += inc * (1.0 if street == 0 else 0.3)
            else:
                decay = (1.0, 0.55, 0.35)[min(max(street - st, 0), 2)]
                if kind == "bet":
                    ka += (P["k_bet"] + P["k_bet_size"] * min(frac, 2.0)) * decay
                elif kind == "raise":
                    ka += (P["k_raise"] + P["k_bet_size"] * min(frac, 2.0)) * decay
                elif kind == "call":
                    k += P["k_call"] * call_w * decay
                else:
                    k -= P["k_check"] * decay
        # Frequent bettors: each bet says less.  Players whose bets keep showing
        # up strong at showdown: each bet says more.
        ka *= clip(0.33 / p.agg, 0.55, 1.6) ** P["agg_gamma"] * \
            clip(_k_of_mean(p.agg_str) / _k_of_mean(AGG_STR_PRIOR), 0.6, 2.5)
        k = max(0.0, k + ka)
        if last is None or last[0] not in AGGR:
            return k, 0.5
        if street == 0:
            return k, clip(0.15 * (1.0 + p.maniac), 0.05, 0.4)
        b = P["b_bet"] if last[0] == "bet" else P["b_raise"]
        b *= clip(p.agg / 0.33, 0.5, 2.0) ** P["agg_gamma"]
        b *= 1.0 - 0.6 * p.nit
        if street == 3:
            b *= clip(p.bluff / 0.30, 0.4, 2.2)
            b += 0.10 * p.bluffer
            b *= self.river_bluff_signal()
        return k, clip(b, 0.02, 0.75)

    def river_bluff_signal(self):
        """>1 when the board and the opponent's discard make busted draws likely."""
        h = self.h
        if h.flop is None or len(h.board) < 5:
            return 1.0
        mult = 1.0
        fs = [0, 0, 0, 0]
        for c in h.flop:
            fs[c & 3] += 1
        bs = [0, 0, 0, 0]
        mask = 0
        for c in h.board:
            bs[c & 3] += 1
            mask |= 1 << (c >> 2)
        fmask = 0
        for c in h.flop:
            fmask |= 1 << (c >> 2)
        flush_draw_flop = max(fs) == 2
        flush_missed = flush_draw_flop and bs[fs.index(2)] < 3
        straight_draw_flop = int(SOUTS[fmask]) >= 1 or len(set(c >> 2 for c in h.flop)) == 3 and \
            (max(c >> 2 for c in h.flop) - min(c >> 2 for c in h.flop)) <= 4
        straight_missed = straight_draw_flop and int(STRAIGHT[mask]) == 0 and int(SOUTS[mask]) == 0
        missed = flush_missed or straight_missed
        if missed:
            mult *= 1.25
        od = h.opp_discard
        if od is not None:
            if missed and (od >> 2) >= 9:             # threw away J+ : kept a draw or a made hand
                mult *= 1.2
            if max(bs) >= 3 and (od & 3) == bs.index(max(bs)):
                mult *= 1.15                          # discarded a card of the flush suit
        return mult

    # ---- equity cache -------------------------------------------------------
    def get_spot(self, street):
        h = self.h
        key = (street, len(h.hole), len(h.board))
        sp = h.spots.get(key)
        if sp is not None:
            return sp
        sc = self.scale_now
        dead = [x for x in (h.my_discard, h.opp_discard) if x is not None and x not in h.hole]
        hole = h.hole
        if len(hole) == 3 and (street >= 2 or len(h.board) > 3):
            kb = np.int64(keys_of(h.board))               # protocol oddity: pick my best two
            sco = [float(keep_score(np.array([keys_of([hole[a], hole[b]])], dtype=np.int64) + kb)[0])
                   for a, b in KEEP_PAIRS]
            a, b = KEEP_PAIRS[int(np.argmax(sco))]
            hole = [hole[a], hole[b]]
        if len(hole) == 3:
            sp = sim_flop(hole, h.board[:3], int(clip(1500 * sc, 120, 5000)), self.rng)
        elif len(h.board) == 3:
            sp = sim_flop2(hole, h.board, dead, h.opp_discard, int(clip(1500 * sc, 120, 5000)), self.rng)
        elif len(h.board) == 4:
            mc = 0 if sc >= 0.45 else int(clip(946 * sc * 2.0, 50, 946))
            sp = enum_turn(hole, h.board, dead, h.opp_discard, h.flop, self.rng, mc)
        else:
            mc = 0 if sc >= 0.08 else 200
            sp = enum_river(hole, h.board[:5], dead, h.opp_discard, h.flop, self.rng, mc)
        h.spots[key] = sp
        return sp

    # ---- discard ------------------------------------------------------------
    def on_discard_request(self, m):
        self.note_time(m)
        h = self.h
        given = m.get("hole_cards", m.get("cards"))
        hc, hr = parse_cards(given)
        if h is None:
            h = self.h = Hand(False, hc, hr, self.sb, self.bb, self.stack)
        elif len(hc) == 3 or (hc and set(hc) != set(h.hole)):
            h.hole, h.raws = hc, hr                 # the engine's view of my hand wins
        board, _ = parse_cards(m.get("board", m.get("community_cards", m.get("board_cards"))))
        if len(board) >= len(h.board):
            h.board = board
        if h.flop is None and len(h.board) >= 3:
            h.flop = h.board[:3]
        # The reply must be one of the cards I hold, written exactly as the
        # engine wrote it: prefer this request's list, then what I track.
        if isinstance(given, (list, tuple)) and len(given) > 0:
            legal = list(given)
        elif h.raws:
            legal = list(h.raws)
        else:
            legal = list(h.dealt)
        card = None
        try:
            idx = self.choose_discard()
            if 0 <= idx < len(h.raws):
                card = h.raws[idx]
        except Exception as e:
            log("discard error:", repr(e))
        if legal and (card is None or card not in legal):
            want = parse_card(card) if card is not None else None
            same = [x for x in legal if want is not None and parse_card(x) == want]
            if same:
                card = same[0]                      # same card, engine's spelling
            else:                                   # fall back to the lowest known card
                known = [x for x in legal if parse_card(x) is not None]
                card = min(known, key=lambda x: parse_card(x) >> 2) if known else legal[0]
        if card is not None:
            h.my_discard = parse_card(card)
        return {"type": "discard", "card": card}

    def choose_discard(self):
        """Index of the hole card to throw away: keep the highest-EV two cards."""
        h = self.h
        hole = h.hole
        if len(hole) != 3:
            return 0 if len(hole) < 3 else len(hole) - 1
        if h.flop is None:                                   # no board: keep the best two
            sc = [float(_pair_score(np.int64(hole[a]), np.int64(hole[b]))) for a, b in KEEP_PAIRS]
            return DISCARD_OF[int(np.argmax(sc))]
        flop = h.flop
        if self.panic():
            kf = np.int64(keys_of(flop))
            sc = [float(keep_score(np.array([keys_of([hole[a], hole[b]])], dtype=np.int64) + kf)[0])
                  for a, b in KEEP_PAIRS]
            return DISCARD_OF[int(np.argmax(sc))]
        n = int(clip(4000 * self.scale_now, 250, 9000))
        spot = sim_flop(hole, flop, n, self.rng)
        h.spots[(1, 3, 3)] = spot
        p = self.opp.profile()
        k, b = self.range_params(1, p)
        ev = np.array(spot.equity(spot.weights(k, b)), dtype=np.float64)
        for j, (a, bb_) in enumerate(KEEP_PAIRS):            # explicit draw / kicker potential
            fd, so, over = draw_features([hole[a], hole[bb_]], flop)
            hi = max(hole[a] >> 2, hole[bb_] >> 2)
            ev[j] += 0.012 * fd + 0.006 * min(so, 2) + 0.003 * over + 0.0004 * hi
        return DISCARD_OF[int(np.argmax(ev))]

    # ---- preflop --------------------------------------------------------------
    def decide_preflop(self, c, p):
        h = self.h
        wm = self.wm
        P = self.P
        _, s = pre_strength(h.hole)                 # percentile among all starting hands
        bb = self.bb
        lag = p.conf > 0 and p.vpip > 0.35 and p.agg > 0.45
        patience = 0.90 * wm.maniac_patience * p.maniac     # versus maniacs: premium hands only
        self.dbg["s"] = s
        if wm.hellmuth > 0 and h.stack0 / (self.sb + self.bb) < 10.0:
            # Short stack (M < 10): push / fold.  Stacks reset to 200 with 1/2 blinds in this
            # format (M = 66), so this only fires if the organisers change the structure.
            if s >= (0.45 if h.button else 0.60):
                return ("raise", c.mx) if c.can_raise else ("call", 0)
            return ("check", 0) if c.to_call <= 0 else ("fold", 0)
        to_call, pot = c.to_call, c.pot
        opp_r = sum(1 for x in h.ev if x[0] == 0 and x[1] == "opp" and x[2] in AGGR)
        my_r = sum(1 for x in h.ev if x[0] == 0 and x[1] == "me" and x[2] in AGGR)
        steal = max(p.nit, p.conf * c01((p.fts - 0.45) / 0.25))          # they give up blinds
        pressure = max(p.maniac, p.conf * c01((p.pf3 - 0.25) / 0.25))    # they re-raise a lot

        # Button, unopened: wide in-position opening range.
        if h.button and opp_r == 0 and my_r == 0 and to_call > 0:
            open_thr = P["open_thr"] * (1.0 - steal) + 0.15 * p.station + 0.12 * pressure
            limp_thr = open_thr if pressure > 0.5 else P["limp_thr"]
            size = P["open_size"] * bb
            if steal > 0.4:
                size = 2.0 * bb                     # relentless cheap steals
            elif p.station > 0.3:
                size = (P["open_size"] + p.station) * bb    # they call anyway: charge more
            if patience > 0:
                open_thr, limp_thr = max(open_thr, patience), max(limp_thr, patience)
            if wm.limp_speculative and s < 0.60 and s >= limp_thr and steal < 0.4 \
                    and is_speculative(h.hole):
                return "call", 0                    # set-mine / draw cheaply in position
            if c.can_raise and s >= open_thr:
                return "raise", size
            if s >= limp_thr:
                return "call", 0
            return "fold", 0

        # Big blind after a limp: raise the top of the range, otherwise see a flop.
        if to_call <= 0:
            thr = P["iso_thr"] - 0.12 * p.station - 0.15 * steal
            size = (P["iso_size"] + p.station - 0.5 * steal) * bb
            if c.can_raise and s >= thr:
                return "raise", size
            return "check", 0

        # Facing a raise: equity against their raising range versus the price.
        if h.pre_spot is None:
            h.pre_spot = sim_preflop(h.hole, int(clip(900 * self.scale_now, 120, 3000)), self.rng)
        spot = h.pre_spot
        k, b = self.range_params(0, p)
        eq = float(spot.equity(spot.weights(k, b)))
        odds = to_call / (pot + to_call)
        committed = (c.stack_left - to_call) <= 0.5 * (pot + to_call)
        nr = opp_r + my_r
        if committed:
            real = 1.0                              # equity realisation
        elif nr >= 2:
            real = P["real3_ip"] if h.button else P["real3_oop"]    # re-raised pots: dominated hands pay off
        else:
            real = P["real_ip"] if h.button else P["real_oop"]
        re_thr = (P["thr_3bet"], P["thr_3bet"], P["thr_4bet"], 0.97)[min(nr, 3)] - 0.08 * p.maniac + 0.04 * p.nit
        if nr == 1 and p.f3b > 0.6:
            re_thr -= 0.15 * p.conf                 # they fold to re-raises: 3-bet lighter
        if nr == 1 and not h.button:                # they open almost any button: 3-bet wider
            re_thr -= P["loose_3bet"] * p.conf * c01((p.pfr - 0.60) / 0.30)
        if lag and s >= 0.97 and not committed and random.random() < wm.trap_frequency:
            return "call", 0                        # under-rep a premium hand against a LAG
        if c.can_raise and s >= re_thr and eq >= 0.52:
            to = h.commit_st + to_call + (pot + to_call)
            if to >= 0.4 * c.mx:
                to = c.mx
            return "raise", to
        raise_bb = (h.commit_st + to_call) / bb
        s_min = 0.0
        if nr == 1 and not h.button:                # tight out of position versus an open
            s_min = clip(P["smin_base"] + P["smin_slope"] * (raise_bb - 2.0), 0.0, 0.55) * (1.0 - 0.5 * p.maniac)
            s_min = min(0.75, s_min + 0.20 * wm.oop_tightness)
        s_min = max(s_min, patience)
        # Survival: do not flip for stacks unless the price is better than 2:1.
        risk = (h.total + to_call) / max(h.stack0, 1.0)
        surv = 0.0 if odds < 1.0 / 3.0 else 0.12 * wm.survival_factor * c01((risk - 0.4) / 0.6)
        self.dbg.update(eq=eq * real, need=odds + 0.01 + surv)
        if eq * real >= odds + 0.01 + surv and s >= s_min:
            return "call", 0
        return "fold", 0

    # ---- postflop -------------------------------------------------------------
    def decide_postflop(self, c, p):
        h = self.h
        spot = self.get_spot(c.street)
        k, b = self.range_params(c.street, p)
        w = spot.weights(k, b)
        if spot.eq.ndim == 2:                       # flop, three cards: best keep option
            eqs = np.asarray(spot.equity(w))
            col = int(np.argmax(eqs))
            e = spot.eq[:, col]
            eq = float(eqs[col])
            kept = [h.hole[i] for i in KEEP_PAIRS[col]] if len(h.hole) == 3 else h.hole[:2]
        else:
            e = spot.eq
            eq = float(spot.equity(w))
            kept = h.hole[:2]
        raw = float(spot.w0 @ e) / max(float(spot.w0.sum()), 1e-12)
        self.dbg.update(eq=eq, k=k, b=b)
        wm = self.wm
        if wm.pot_control_threshold < 1.0 or wm.hero_fold_sensitivity > 0 \
                or wm.check_back_medium > 0 or wm.nit_strict_fold > 0 or wm.hellmuth > 0:
            c.tier = made_tier(kept, h.board)       # only the White Magic rules need it
        if c.to_call > 0:
            # Calibration: against this opponent, how far have my estimates been from the
            # showdown results of the turn / river calls I made?  Shift by the shrunk bias.
            P = self.P
            eq_used = eq
            if P["cal_gain"] > 0 and c.street >= 2:
                bias = clip(p.cal_d / (p.cal_n + P["cal_n0"]), -P["cal_cap_lo"], P["cal_cap"])
                eq_used = eq - P["cal_gain"] * bias
            r = self.facing_bet(c, p, spot, w, e, eq_used, raw, kept)
            h.last_face = (c.street, eq, r[0])
            return r
        return self.no_bet(c, p, spot, w, e, eq, kept)

    def no_bet(self, c, p, spot, w, e, eq, kept):
        """Checked to me (or first to act): bet when it beats checking in EV."""
        h = self.h
        pot, street, ip = c.pot, c.street, h.ip
        river = len(h.board) >= 5
        if not c.can_raise or pot <= 0:
            return "check", 0
        wet = board_wetness(h.board)
        wm = self.wm
        P = self.P

        # Maniacs: trap with monsters out of position and let them bet for us.
        if p.maniac >= 0.4 and eq >= 0.80 and not ip:
            induce = 0.8 if not river else clip(p.river_agg + 0.15, 0.3, 0.85)
            if random.random() < induce:
                h.trap = True
                return "check", 0
        # Pot control: one pair in position on a wet board checks back to keep the
        # pot small and to induce.
        if wm.check_back_medium > 0 and ip and not river and c.tier == 1 and wet >= 0.5 \
                and eq < 0.75 and p.station < 0.4 and random.random() < wm.check_back_medium:
            return "check", 0

        ftb = p.ftb
        if P["mk_gain"] > 0 and p.conf > 0:         # fold chance from the action transition matrix
            prev = 0
            for x in h.ev:
                if x[1] == "opp" and x[0] >= 1 and x[2] in MK_STATE:
                    prev = MK_STATE[x[2]]
            ftb = (1.0 - P["mk_gain"]) * ftb + P["mk_gain"] * self.opp.markov_fold(prev)
        f0 = ftb * (0.90, 0.90, 1.00, 1.08)[min(street, 3)]
        if any(x[1] == "opp" and x[2] in AGGR and 1 <= x[0] < street for x in h.ev):
            f0 *= 0.85                              # they showed strength earlier
        if ip:
            f0 *= 1.08                              # they checked to me
        if p.station >= 0.4:
            sizes = wm.station_sizes or (0.75, 1.0, 1.25)   # thin, fat value
            if wm.hellmuth > 0:
                sizes = (0.75, 0.80)
        elif p.nit >= 0.4:
            sizes = (0.33, 0.5)                     # small, frequent stabs
            if wm.hellmuth > 0 and wet < 0.5:
                sizes = (0.30, 0.40)                # probe bets on dry boards
        elif wet >= 0.5:
            sizes = P["sizes_wet"]
        elif wm.small_ball > 0:
            sizes = (0.30, 0.40)                    # small ball on dry boards
        else:
            sizes = P["sizes_dry"]
        if river:
            sizes = tuple(sizes) + tuple(P["sizes_river"])
        best = None
        for sz in sizes:
            to = clip(h.commit_st + sz * pot, c.mn, c.mx)
            B = to - h.commit_st
            if B <= 0:
                continue
            f = clip(f0 * (P["fold_base"] + P["fold_slope"] * B / pot), 0.02, 0.92)
            eqc = spot.vs_top(w, e, 1.0 - f)        # equity when called
            ev = f * pot + (1.0 - f) * (eqc * (pot + 2.0 * B) - B)
            if best is None or ev > best[0]:
                best = (ev, to, eqc)
        if best is None:
            return "check", 0
        ev, to, eqc = best
        gain = ev - eq * pot * (1.0 if ip else 0.95)

        # Out of position against the previous street's aggressor: check to them
        # unless the hand is strong enough to welcome a raise.
        prev_aggr = [x for x in h.ev if x[0] == street - 1 and x[2] in AGGR]
        if not ip and street >= 2 and prev_aggr and prev_aggr[-1][1] == "opp" and eqc < 0.60:
            return "check", 0

        # Pot control: a one-pair hand only builds the pot up to its share of the stack.
        if wm.pot_control_threshold < 1.0 and c.tier == 1 and street >= 2 and p.station < 0.4 \
                and sum(1 for x in h.ev if x[0] == 0 and x[2] in AGGR) <= 1:
            room = wm.pot_control_threshold * h.stack0 - h.total
            if to - h.commit_st > room:
                to = h.commit_st + room
                if to < c.mn:
                    return "check", 0

        if eqc >= P["val_thr"]:                     # value bet
            if gain > P["val_gain"] * pot or eqc >= P["val_sure"]:
                if eqc >= 0.58 and (c.stack_left <= 1.4 * pot or to >= 0.7 * c.mx):
                    to = c.mx                       # low SPR: get it in
                return "raise", to
            return "check", 0

        # Bluffs and semi-bluffs, mixed so the betting range stays balanced.
        draw = (not river) and eq >= 0.30
        if p.station >= 0.4 and (not draw or wm.hellmuth > 0):
            return "check", 0                       # never pure-bluff a calling station
        cbet = street == 1 and h.pf_agg == "me"
        if gain <= P["bluff_gain"] * pot and not (cbet and p.nit >= 0.4):
            return "check", 0
        freq = P["bluff_draw"] if draw else (P["bluff_ip"] if ip else P["bluff_oop"])
        freq = freq * (1.0 - 0.8 * p.station) + 0.4 * p.nit
        if cbet:
            freq += 0.15 + 0.30 * p.nit
        if random.random() < freq:
            if not draw:                            # pure bluffs risk the smallest size
                to = min(to, max(c.mn, h.commit_st + sizes[0] * pot))
            return "raise", to
        return "check", 0

    def facing_bet(self, c, p, spot, w, e, eq, raw, kept):
        """Call when equity against their betting range beats the pot odds."""
        h = self.h
        pot, to_call, street = c.pot, c.to_call, c.street
        river = len(h.board) >= 5
        odds = to_call / (pot + to_call)
        allin_call = to_call >= c.stack_left - 0.5
        last = next((x for x in reversed(h.ev) if x[1] == "opp" and x[0] == street), None)
        fd = so = over = 0
        if not river and len(kept) == 2:
            fd, so, over = draw_features(kept, h.board)

        P = self.P
        need = odds + (P["need_river"] if river else P["need_early"])
        if not river and not allin_call:
            need -= 0.035 * fd + 0.015 * min(so, 2)          # implied odds of draws
        need -= 0.04 * p.bluffer * (1.0 if river else 0.5)   # call bluffers down lighter
        need -= 0.03 * p.maniac

        # Nits: a big bet or any raise is the goods. Fold everything but the near-nuts.
        big = last is not None and (last[2] == "raise" or last[3] >= 0.9)
        if p.nit >= 0.4 and big and odds > 0.15 and raw < 0.92:
            return "fold", 0

        # ---- White Magic adjustments (all neutral unless their knob is set) ----
        wm = self.wm
        single_raised = sum(1 for x in h.ev if x[0] == 0 and x[2] in AGGR) <= 1
        resist = last is not None and (last[2] == "raise" or last[3] > 0.65)
        if wm.lag_dry_call > 0:                     # aggressive players: call wider on dry boards
            need -= 0.04 * wm.lag_dry_call * (1.0 - board_wetness(h.board)) * max(p.maniac, p.bluffer)
        if wm.survival_factor > 0 and odds >= 1.0 / 3.0:     # no thin calls for most of the stack
            risk = (h.total + to_call) / max(h.stack0, 1.0)
            need += 0.12 * wm.survival_factor * c01((risk - 0.4) / 0.6)
        if wm.hero_fold_sensitivity > 0 and street >= 2 and resist:
            passive = p.conf * c01((0.29 - p.agg) / 0.08)    # full strength below ~21% aggression
            eq *= 1.0 - wm.hero_fold_sensitivity * passive * (1.0 if c.tier <= 1 else 0.3)
        if (wm.nit_strict_fold > 0 or wm.hellmuth > 0) and p.nit >= 0.4 and odds > 0.15 and c.tier < 2 and last is not None \
                and (last[2] == "raise" or (street >= 2 and last[3] >= 0.6)):
            return "fold", 0                        # a nit resists: two pair or better, or out
        if wm.pot_control_threshold < 1.0 and c.tier == 1 and street >= 2 and single_raised \
                and max(p.maniac, p.bluffer) < 0.4 \
                and h.total + to_call > wm.pot_control_threshold * h.stack0:
            return "fold", 0                        # never go broke with one pair
        self.dbg.update(eq=eq, need=need)

        thr = P["raise_thr"] - (0.04 if h.raises_st >= 3 else 0.0)   # at the raise cap nobody can come back
        if c.can_raise and eq >= thr and spot.vs_top(w, e, 0.5) >= P["raise_top"]:
            deep = (c.stack_left - to_call) > 0.75 * (pot + to_call)
            lag = p.conf > 0 and p.vpip > 0.35 and p.agg > 0.45
            if not river and deep and (
                    (p.maniac >= 0.4 and (h.trap or random.random() < 0.8))
                    or (lag and raw >= 0.97 and random.random() < wm.trap_frequency)):
                return "call", 0                    # keep them betting; raise the river
            sz = 1.0 if board_wetness(h.board) >= 0.5 else 0.75
            to = h.commit_st + to_call + sz * (pot + to_call)
            if to >= 0.6 * c.mx or (c.stack_left - to_call) <= 0.9 * (pot + to_call):
                to = c.mx
            return "raise", to
        if eq >= need:
            return "call", 0
        # Occasional semi-bluff raise with a strong draw against players who can fold.
        if (not river and c.can_raise and h.raises_st <= 1 and p.station < 0.3 and p.maniac < 0.3
                and (wm.lag_dry_call <= 0 or p.bluffer < 0.3)        # never re-bluff a bluffer
                and fd and (so >= 1 or over >= 1) and eq >= 0.30
                and random.random() < 0.15 + 0.25 * p.nit):
            return "raise", h.commit_st + to_call + 0.8 * (pot + to_call)
        return "fold", 0


# ---------------------------------------------------------------------------
# 6. Precomputed preflop table and main loop
# ---------------------------------------------------------------------------
# Equity (x1000, 3 digits each) of every 3-card starting-hand class against a
# random hand, in canon_classes() order.  Generated offline with sim_preflop.
PRE_EQ_DATA = (
    "319393409410414355400416336342329332313427438439443441450390406416347355346341323430445364370360"
    "356338358365361356330461470472474473480479478426408417354360352349329438444368376365366346367375"
    "367365349464474383392381383364389388382385360386385379379366494497507507507508506510510511462413"
    "426362364359356335441451373378375369354381378374379355472475388395389387376401399391395376403400"
    "395394374499506401406397396380402409400402385408409401402383401406405399384526530537537536544541"
    "541545542543539504419430366371365366347449452380384379376356386394386388371472482398403397392381"
    "406411397406387405411410411390501510406410404402386411416410408395416422414410399419422419417401"
    "536535423427417418402424432418421407425433423423408425433424423404427430421424406567568572567576"
    "576577572572573582575578580539422439381381383376358448460386389382382368397400393397374475485409"
    "410405399383413419408414397421422422417396509513414424410406389420425417417401427435423425409432"
    "437430432415536544425432425420407434439430424411438448435431419443444441439423441450440439424573"
    "573443447445437422447452443445430450458446447434449457449446433448454448447434449457451447434595"
    "599611607612606609612612613613611609610615613577429437398401393387370455467403406403400383410417"
    "411410396482489408412409408391419422417414400432430426427411514522422426415412397426434425426410"
    "440441438434421442448445444425546548432433431424413441445438437423445450448446428454461452449439"
    "462465457459440575575449454450445434454456450448434461463461454445464472459464450470473465469451"
    "469475466466452600606468473460462445471475472465455474478470467460481483478473468481483478474462"
    "484485479481461478487482480463633634645644648645649645649646654648650650649645651648619430451409"
    "417411405384459471418422421413397425428428424408491501430431424424408434439435433410447446446442"
    "423517527429430427424404436439435430415448443442443425450461452449437547549439446437434420445451"
    "445445430455459459451436468474463462450465474473469454573578457461454454439463468463454442468470"
    "465461451479476478472462479484480480465484489482483469604610475482469468460482484474479461486491"
    "481477464488494484489470493501491491474497502499496482497504497496481637639496499497491481498504"
    "496491487502510504499492507513507499493509516512506495514521507513497511520511514497510518511510"
    "497669671680677683680684679684677685675682677683676684682681679652439456425429422422406463478432"
    "441434427415441444440438417491505445450440437428454458449450429459465459455446526529445456448442"
    "427456459453447432460470457460444468475469465453551557446454453443435457465454447440464470461463"
    "446470479475472449480485488477469583581469475466462450475473473464454478483474474461485487487485"
    "471495500495487475498506498502485606614487493486479474488494488481477493498494487479496506497494"
    "488505512504499491512514511510499515522513513501641645504514504501490509515508505498515522510508"
    "497516523516514502524531520520507527533529526513529539533529517531538527533517676673514523509508"
    "501520527519511506523528524517513527533525526515529535529526517534542532535525533540533539520535"
    "542536533521534543536534519704704711707713711712709716709715713714714722712714711716713711713687"
    "441467441447445439426472494448458454443433458460458450439499511457468462456442467471462464450476"
    "480476469461532540469474468459451474477469470456478490477480459486496489482472554560475483473467"
    "459483485477467463488494485479468492498494492481499506498500482583594481488484471466487490485475"
    "468492498489487476494502501491481506511505502489510520514511497616620501507496494482507513503494"
    "484513517505502490512518514506499518524517514501523530527526508530539533531518646646519528517513"
    "503522533523516511526534528523513534537530526519540542535536521543550537542529549551551547537550"
    "558551554538677682523537527523511530542532528521536544536533521543545543535522547550544538533547"
    "556547546536553562554551538556563557556543560559559554542708705536551533536523542550544538532545"
    "554546543535555559553552541556561558555545559564558559543565568563561551566575562561549563570564"
    "562551570577569569557735737747739748742746742750749750743750752751745743745750745749745752746726"
    "448475471484471462456475498477491483470463488488484474467504524493497492480470493499494483479495"
    "509500493479538550494507499486481502509503495481504514511500490515521517506505560576508515506497"
    "489515520512503495517524517506502525526528521508523532524522510594600516526518511501518528521515"
    "507528532531520507532536531527514532534530529518539543540532523620623525531525515513531535527521"
    "515535539532527520535547542536528540547539531524542548546540532550556555546541651652543549544539"
    "527553560547545536555560552546540557569562552547557562556553548566567561556550570576565566556574"
    "582575573566687688548557553544540559563558549543562571558559553573575568560557565577570564554575"
    "575574569562581579581573563579587584580569584593587578568715714561571559554548566578574560554570"
    "580574571560577589579574566580586577570564582589585577572584595583584576593596591587576586598591"
    "592582596601598592585739738569578567563555579585576573564582589581579570585595589583575588594585"
    "579578588599595589588595604600593589597606603598590601608602600592603610604602595609615615610594"
    "774773782782784785784780787787786785787784787784788785788786785784787785785782763"
)

load_pre_table(PRE_EQ_DATA)


def _fallback_reply(typ, m, bot):
    """Last-resort legal reply if a handler raised."""
    if typ == "discard_request":
        cards = m.get("hole_cards")
        if not isinstance(cards, (list, tuple)) or not cards:
            cards = (bot.h.raws or bot.h.dealt) if bot.h is not None else None
        return {"type": "discard", "card": cards[0] if cards else None}
    names, _, _, _ = parse_legal(m)
    tc = num(m, ("to_call", "call_amount", "amount_to_call"), 0.0)
    if "check" in names or (tc <= 0 and "fold" not in names):
        return {"type": "action", "action": "check"}
    return {"type": "action", "action": "fold" if ("fold" in names or not names) else "call"}


def main():
    bot = Player()
    stdin, stdout = sys.stdin, sys.stdout
    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        t0 = time.perf_counter()
        try:
            m = json.loads(line)
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        typ = str(m.get("type", "")).lower()
        reply = None
        try:
            if typ == "action_request":
                reply = bot.on_action_request(m)
            elif typ == "discard_request":
                reply = bot.on_discard_request(m)
            elif typ == "new_hand":
                bot.on_new_hand(m)
            elif typ == "discard_result":
                bot.on_discard_result(m)
            elif typ == "hand_result":
                bot.on_hand_result(m)
            elif typ == "init":
                bot.on_init(m)
            elif typ == "match_end":
                break
            else:
                bot.on_other(m)                     # unknown types are otherwise ignored
        except Exception as e:
            log("handler error (%s): %r" % (typ, e))
            if typ in ("action_request", "discard_request"):
                try:
                    reply = _fallback_reply(typ, m, bot)
                except Exception:
                    reply = {"type": "action", "action": "fold"} if typ == "action_request" \
                        else {"type": "discard", "card": None}
        if reply is not None:
            stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            stdout.flush()
        dt = (time.perf_counter() - t0) * 1000.0
        bot.used_ms += dt
        bot.hand_ms += dt


if __name__ == "__main__":
    main()
