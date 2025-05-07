# esch

`esch` is a layered visualization library, supporting SVG animation.

```python
import esch
import numpy as np

x = np.random.randn((3, 100, 100))  # <- init data
e = esch.init(x.shape)              # <- make a plot object with shape (small_multiples x height x width)
e = esch.tile(e, x)                 # <- add data to the e object
```

`esch` is best used by making a function constructuor

```python
def tile_fn(x):
    return esch.tile(esch.init(x.shape), x)
```

TODO:

0. [ ] sims stuff variable pos.
1. [ ] Default font size to 12pt across sizes
2. [x] Add mesh plot for
3. [ ] Known issue is that for animation, fill is determined by first value
       (and will thus not flip if sign changes during animation).
4. [ ] Logic to infer if we have animation is dump.
       Assumes one dim is 20x times another. Many times that is not the case
5. [ ] Minimise svg size with rounded floats.
