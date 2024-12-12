# JAX Automatic Differentiation Implementation

A shot at implementing automatic differentiation from scratch using JAX in my free time. 

## Setup

1. Create a virtual environment using conda:
```bash
conda create --name jax-grad python=3.10
conda activate jax-grad
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```plaintext
jax-grad/
├── jax_grad/                  
│   ├── core/                  
│   │   ├── __init__.py
│   │   └── autodiff.py        
│   ├── ops/                   
│   │   ├── __init__.py
│   │   └── math.py           
│   ├── tensor/               
│   │   ├── __init__.py
│   │   └── ops.py           
│   ├── utils/
│   │   ├── __init__.py
│   │   └── grad_check.py    
│   └── viz/                  
├── tests/                    
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── examples/
│   └── basic_usage.py
├── docs/
├── setup.py
├── requirements.txt
└── README.md
```

## Features

- Forward-mode automatic differentiation
- Basic mathematical operations support
- Test suite for verification

## Usage

Basic example:
```python
from jax_grad import grad

def f(x):
    return x ** 2

df = grad(f)
result = df(3.0)  # Should return 6.0
```

## Roadmap

For detailed implementation roadmap and future plans, please see [ROADMAP.md](docs/ROADMAP.md)
