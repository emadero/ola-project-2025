# Online Learning Applications Project 2025

## Project Overview

This project implements online learning algorithms for dynamic pricing of multiple product types under production constraints. The goal is to design algorithms that can adaptively set prices to maximize revenue while respecting inventory limitations.

## Problem Setting

A company must choose prices dynamically for multiple products:
- **T** rounds of interaction
- **N** types of products  
- **P** discrete set of possible prices
- **B** production capacity constraint

### Buyer Behavior
- Each buyer has valuations for each product type
- Buys all products priced below their respective valuations

## Team Structure

### Person 1: Federico Madero - Single Product Specialist
**Responsibilities:**
- Requirement 1: Single Product & Stochastic Environment
- Requirement 3: Best-of-Both-Worlds (Single Product)
- Base environments for team extension

### Person 2: Maxence Guyot - Multiple Products Specialist  
**Responsibilities:**
- Requirement 2: Multiple Products & Stochastic Environment
- Requirement 4: Best-of-Both-Worlds (Multiple Products)

### Person 3: Amirhassan Darvishzade - Analysis & Comparison Specialist
**Responsibilities:**
- Requirement 5: Slightly Non-Stationary Environments
- Performance analysis and visualization
- Final comparisons and presentation

## Project Structure

```
online-learning-applications-project-2025/
├── README.md
├── requirements.txt
├── main.py
├── environments/          # Testing environments
│   ├── __init__.py        # Base environment classes
│   ├── stochastic.py      # Stochastic environments (Person 1)
│   ├── non_stationary.py  # Highly non-stationary (Person 1)
│   └── slightly_ns.py     # Slightly non-stationary (Person 3)
├── algorithms/            # Algorithm implementations
│   ├── __init__.py        # Base algorithm classes
│   ├── single_product/    # Single product algorithms
│   │   ├── ucb.py
│   │   ├── constrained_ucb.py
│   │   └── primal_dual.py
│   └── multiple_products/  # Multiple product algorithms
│       ├── combinatorial_ucb.py
│       ├── sliding_window.py
│       └── primal_dual.py
├── experiments/          # Experiment scripts
│   ├── requirement1.py  #  Req 1
│   ├── requirement2.py  #  Req 2
│   ├── requirement3.py  #  Req 3
│   ├── requirement4.py  #  Req 4
│   └── requirement5.py  #  Req 5
├── requirement_5_files/               # Files to execute requirement 5
│
├── results/             # Experiment results
│   ├── figures/         # Generated plots
│   └── data/            # Numerical results
└── presentation/        # Final presentation materials
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/emadero/ola-project-2025
cd online-learning-applications-project-2025
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Experiments
```bash
# Single Product experiments
python experiments/requirement1.py
python experiments/requirement3.py

# Multiple Product experiments  
python experiments/requirement2.py
python experiments/requirement4.py

# Slightly Non-Stationary Environments experiments
python experiments/requirement5.py
```

### Running All Experiments
```bash
python main.py
```

## Requirements Implementation

### Requirement 1: Single Product & Stochastic Environment
- [ ] Stochastic environment with single product valuations
- [ ] UCB1 algorithm ignoring inventory constraints
- [ ] UCB1 algorithm with inventory constraints

### Requirement 2: Multiple Products & Stochastic Environment  
- [ ] Joint distribution over multiple product valuations
- [ ] Combinatorial UCB1 ignoring inventory constraints

### Requirement 3: Best-of-Both-Worlds (Single Product)
- [ ] Highly non-stationary environment
- [ ] Primal-dual method with inventory constraints

### Requirement 4: Best-of-Both-Worlds (Multiple Products)
- [ ] Highly non-stationary environment with correlated valuations
- [ ] Primal-dual method for multiple products

### Requirement 5: Slightly Non-Stationary Environments
- [ ] Slightly non-stationary environment (interval-based)
- [ ] Combinatorial UCB with sliding window
- [ ] Performance comparison analysis

## Key Design Decisions

### Environment Interface
All environments inherit from `BaseEnvironment` and implement:
- `reset()`: Initialize environment
- `step(selected_prices)`: Execute one round
- `get_buyer_valuations()`: Generate buyer valuations

### Algorithm Interface  
All algorithms inherit from `BaseAlgorithm` and implement:
- `select_prices()`: Choose prices for current round
- `update(prices, rewards, buyer_info)`: Learn from feedback

### Constraint Handling
The `ConstraintHandler` class provides utilities for:
- Enforcing production capacity constraints
- Calculating constraint violation costs

## Coordination Guidelines

### Weekly 
- **Week 1**: Foundation (environments + basic algorithms)
- **Week 2**: Core algorithms implementation
- **Week 3**: Advanced methods (primal-dual)
- **Week 4**: Integration, testing, and presentation


### Communication
- Weekly sync meetings to ensure compatibility
- Shared documentation of data formats
- Code reviews before major integrations

## Results and Analysis

Results will be saved in:
- `results/data/`: Numerical results as CSV files
- `results/figures/`: Generated plots and visualizations

Key metrics to track:
- Cumulative regret over time
- Revenue comparison vs oracle
- Algorithm convergence rates
- Constraint satisfaction rates

## Deliverables

1. **Code Repository**: Complete implementation with all requirements
2. **Presentation Slides**: 20-minute presentation covering:
   - High-level algorithm details
   - Empirical results with graphs
   - Discussion of unexpected results
3. **Documentation**: Clear usage instructions and API documentation

## Timeline

- **Deadline**: July 11th, 2025
- **Presentations**: Days following submission
