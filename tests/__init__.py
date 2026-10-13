from hypothesis import settings

# Property tests render SVGs; under parallel workers one example can take
# longer than Hypothesis's default 200 ms deadline without anything being wrong.
settings.register_profile("esch", deadline=None)
settings.load_profile("esch")
