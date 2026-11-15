# Benchmark report — 12 November 2026

## Setup

- **Bots:** vidhu (`bot1.py`, trained equity engine), anirudh (`bot.py`), rohit (`Rohit's_bot.py`), and the 23 sparring bots in `OPPS/` (18 original, 5 added today: `x_value`, `x_3bet`, `x_checkraise`, `x_sticky`, `x_barrel`).
- **Format:** round robin, every pair plays one Stage 1 match: two legs of 200 hands over the same deals with hole cards and seats swapped, fresh bots for each leg. 325 matches, 130,000 hands.
- **Engine:** `benchmark/engine2.py`, written from Protocol Specification 1.0. It is not the organisers' `engine.py`.
- **How they ran:** vidhu, anirudh and rohit as real `python` subprocesses over stdin/stdout; sparring bots in-process.
- **Reproduce:** `python benchmark\bench.py . tournament out.json 1 200 11`, then `python benchmark\report2.py out.json standings`.

## Standings

Ranked by match points, then by net chips per hand capped at ±400 per match (Rulebook 8.3).

| # | Bot | W-D-L | Net chips | bb/100 | Capped chips/hand |
|---|---|---|---|---|---|
| 1 | **vidhu** | 25-0-0 | +162,035 | +810 | +0.980 |
| 2 | **anirudh** | 24-0-1 | +178,339 | +892 | +0.940 |
| 3 | h_polar | 21-0-4 | +78,421 | +392 | +0.680 |
| 4 | x_value | 21-0-4 | +88,256 | +441 | +0.650 |
| 5 | nit | 21-0-4 | +33,905 | +170 | +0.584 |
| 6 | shover | 20-0-5 | +59,470 | +297 | +0.617 |
| 7 | x_sticky | 19-0-6 | +38,567 | +193 | +0.541 |
| 8 | tag | 16-0-9 | +41,884 | +209 | +0.270 |
| 9 | eqbot | 15-0-10 | +39,202 | +196 | +0.215 |
| 10 | **rohit** | 15-0-10 | +38,493 | +193 | +0.191 |
| 11 | trapper | 14-0-11 | +35,320 | +177 | +0.139 |
| 12 | h_float | 13-0-12 | +29,054 | +145 | +0.074 |
| 13 | balanced | 13-0-12 | +39,365 | +197 | +0.060 |
| 14 | x_checkraise | 12-0-13 | +33,742 | +169 | -0.021 |
| 15 | x_3bet | 12-0-13 | +31,229 | +156 | -0.042 |
| 16 | h_adapt | 11-0-14 | +43,171 | +216 | -0.089 |
| 17 | x_barrel | 11-0-14 | +30,748 | +154 | -0.097 |
| 18 | station | 8-1-16 | -52,715 | -264 | -0.359 |
| 19 | lag | 8-0-17 | -5,773 | -29 | -0.360 |
| 20 | bluffer | 8-0-17 | -28,079 | -140 | -0.391 |
| 21 | maniac | 6-0-19 | -93,167 | -466 | -0.520 |
| 22 | riverjam | 4-0-21 | -157,484 | -787 | -0.680 |
| 23 | minbet | 4-0-21 | -202,833 | -1,014 | -0.687 |
| 24 | allin | 1-2-22 | -174,396 | -872 | -0.840 |
| 25 | starter | 1-1-23 | -99,801 | -499 | -0.855 |
| 26 | random_bot | 0-0-25 | -186,953 | -935 | -1.000 |

## Head-to-head between the three main bots

Ten extra duplicate matches per pairing (4,000 hands each) on fresh deals.

| Pairing | Matches won | Average per match | bb/100 (± standard error) |
|---|---|---|---|
| vidhu vs anirudh | 4 – 6 | vidhu −30 chips | −3.8 ± 16.9 |
| vidhu vs rohit | 10 – 0 | vidhu +8,591 chips | +1,074 ± 43 |
| anirudh vs rohit | 10 – 0 | anirudh +7,256 chips | +907 ± 47 |

vidhu and anirudh are statistically level. Before today's training vidhu lost this pairing 7–33 at −69 ± 9 bb/100.

## vidhu, anirudh and rohit against each opponent (tournament match, bb/100)

| Opponent | vidhu | anirudh | rohit |
|---|---|---|---|
| vidhu | – | −25 | −1,006 |
| anirudh | +25 | – | −996 |
| rohit | +1,006 | +996 | – |
| nit | +200 | +219 | −26 |
| x_value | +263 | +666 | −413 |
| balanced | +295 | +513 | +186 |
| shover | +309 | +408 | −303 |
| tag | +418 | +406 | −103 |
| x_checkraise | +426 | +757 | +5 |
| h_polar | +500 | +662 | −77 |
| eqbot | +541 | +699 | +73 |
| x_barrel | +657 | +762 | −176 |
| trapper | +791 | +642 | +51 |
| x_3bet | +812 | +1,001 | +43 |
| bluffer | +821 | +525 | +587 |
| h_adapt | +887 | +627 | −34 |
| lag | +917 | +717 | +384 |
| h_float | +957 | +1,465 | +158 |
| maniac | +1,025 | +1,308 | +1,055 |
| minbet | +1,150 | +1,614 | +1,712 |
| x_sticky | +1,175 | +1,233 | −353 |
| riverjam | +1,244 | +1,504 | +1,461 |
| allin | +1,273 | +1,003 | +970 |
| station | +1,365 | +1,896 | +151 |
| random_bot | +1,422 | +836 | +1,108 |
| starter | +1,778 | +1,860 | +354 |

Each cell is a single 400-hand match, so individual cells are noisy; the standings and head-to-heads are the reliable numbers.

## Playing style over the tournament

| Bot | Hands played % | Preflop raise % | Postflop bets per call | Saw flop % | Went to showdown % | Won at showdown % |
|---|---|---|---|---|---|---|
| vidhu | 65 | 48 | 2.06 | 74 | 56 | 62 |
| anirudh | 69 | 54 | 3.22 | 82 | 48 | 64 |
| rohit | 66 | 15 | 1.82 | 90 | 76 | 51 |

## Reliability (90 legs each, as subprocesses)

| Bot | Time bank used, average | Maximum | Engine corrections | Crashes | Exit code |
|---|---|---|---|---|---|
| vidhu | 7.5 s | 10.8 s of 30 s | 0 | 0 | 0 |
| anirudh | 6.9 s | 10.2 s of 30 s | 0 | 0 | 0 |
| rohit | 9.3 s | 13.4 s of 30 s | 0 | 0 | 0 |

Timings were taken with eleven matches running in parallel on a 12-core laptop, so they overstate single-match use. vidhu's stderr stayed under 750 bytes per leg (limit 64 KB).

## What changed in vidhu today

Tuned against anirudh on duplicate deals (30 matches per variant, fresh deals each round) and checked against the whole sparring pool:

| Setting | Before | After |
|---|---|---|
| Bet sizes on dry boards (pot fractions) | 0.5, 0.75 | 1.0, 1.5 |
| Bet sizes on wet boards | 0.66, 1.0 | 1.25, 2.0 |
| Bluff frequency in / out of position | 0.50 / 0.38 | 0.70 / 0.55 |
| Minimum edge required to bluff | 3% of pot | none |
| Big-blind calling range against an open | wider | tighter |

Tested and rejected (neutral or worse): larger sizes still, a mixed size menu, more bluffing still, calling more or less, thinner value bets, wider opening, no limping, wider 3-betting, showdown calibration of calls, reading frequent bettors as stronger or weaker, and every "White Magic" rule.

## Limits of this benchmark

- The engine is my implementation of the specification; a run on the organisers' `engine.py` is still needed before upload.
- anirudh is the only opponent strong enough to separate strategies, so vidhu's tuning rests on one bot. The gain did not cost anything against the other 24.
- Stage 2 (500-hand legs) was not part of this run.
