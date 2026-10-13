# All tests in parallel; also regenerates the visual gallery (visual/index.html).
test *args:
    uv run pytest {{ args }}

# Lint and dead-code checks.
check:
    ruff check esch main.py tests
    vulture esch main.py --min-confidence 80

# Size and the most complex functions (a report: complexity above 15 does not fail it).
map:
    scc --no-cocomo esch main.py tests
    -complexipy esch main.py --sort desc --top 15

chat:
    lila --codex --write --network --session $(basename "$PWD")
