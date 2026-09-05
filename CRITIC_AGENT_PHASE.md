# Critic Agent phase

Implemented against Q102-Q118.

Key guarantees:
- structural draft errors raise `CriticInputValidationError` before semantic review
- four checks are always evaluated independently by the LLM
- verdict is deterministically derived from the four checks
- issues use `[RESEARCH]`, `[WRITING]`, `[ANALYSIS]`, `[INFO]`
- per-check issue category compatibility is deterministically validated
- correction target is derived as `research` > `writing` > `analysis` > `none`
- Critic validation retries are capped at two after the initial attempt
- Critic does not repair the report
