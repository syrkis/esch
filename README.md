## Usage as cli

Run esch with your Typst file as an argument:

```bash
npx @syrkis/esch your-file.typ
```

## Usage as a library

```python
import esch
import numpy as np

x = np.random.rand((100, 10, 10))
dwg = esch.hinton(x)
x.saveas('hinton.svg')
```
