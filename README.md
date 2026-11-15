# vidhu — a Discard Hold'em bot

A bot for the Pokerbots competition (heads-up no-limit Hold'em with three hole cards and a
simultaneous exposed discard after the flop). Python 3.11, standard library plus numpy, one file.

**Keep this repository private until the competition results are published.** The rulebook
prohibits sharing code or strategy between teams.

## Files

| Path | What it is |
|---|---|
| `bot.py` | The bot, exactly as submitted. |
| `BENCHMARK.md` | Tournament report from 12 November 2026 (26 bots, 325 matches). |
| `benchmark/` | Local test harness: a protocol-1.0 engine, duplicate-deal match runner and report scripts. |
| `OPPS/` | Sparring bots used for testing (calling station, maniac, nit, bluffer and others). `starter.py` and `random_bot.py` come from the organisers' starter kit. |
| `history/` | Earlier versions kept for reference; none of them is the submission. |

Two other people's bots that were used as strong opponents during testing are not included, and
neither is the organisers' starter kit (engine and PDFs).

## How the bot works

1. **Hand evaluator** — vectorised numpy bitmask evaluation of 5 to 8 cards.
2. **Equity engine** — Monte Carlo before and on the flop (one equity column per possible
   discard), exact enumeration on the turn and river.
3. **Opponent model** — statistics collected within a leg only (VPIP, preflop raise, aggression,
   fold-to-bet, showdown strength) turned into calling-station / maniac / nit / bluffer scores.
   The first 20 hands are played on priors.
4. **Decisions** — the opponent's range is weighted by the actions they took; bets are chosen
   by expected value, calls by equity against that range versus pot odds.
5. **Time bank** — the rollout count scales with the time left, and the bot switches to
   constant-time play if the bank is nearly empty.

All tunable constants are in `PARAMS` and the `WhiteMagic` class in `bot.py`. Each value was set
by A/B testing on duplicate deals; options that tested neutral or worse are still in the code
but switched off.

## Running it

Against the organisers' engine (from the starter kit):

    python engine.py bot.py example_bots/random_bot.py
    python engine.py bot.py example_bots/random_bot.py --stage 2

With the local harness (it expects the bot as `bot1.py` in the project folder, so copy
`bot.py` to `bot1.py` first):

    python benchmark/bench.py . versus out.json vidhu all 2 200 8
    python benchmark/report2.py out.json focus vidhu

The `anirudh` and `rohit` entries in `benchmark/bench.py` refer to the two bots that are not
in this repository; the `tournament` mode needs them, the `versus` mode above does not.

## Results before submission

- Organisers' engine: 11 full matches (Stage 1 and Stage 2), no notices, no forfeits, at most
  14.8 s of time used in a Stage 2 match (75 s bank per leg).
- Local harness: wins every match against all 23 sparring bots; level with the strongest
  available opponent over 90 matches (−2.5 bb/100, within noise).
