# Capstone Proposal
## FCPS Waste Cost Reduction with Contextual Multi-Armed Bandits
### Authors: Neeraj Shashikant Magadum, Varun Gholap
### Advisor: Prof. Amir Jafari
#### The George Washington University, Washington DC
#### M.S. in Data Science – Fall 2025

## Overview
        This project develops an adaptive recommendation and decision-support system
        for reducing food waste in K–12 school cafeterias using Contextual Multi-Armed
        Bandits (CMAB). Using the LinUCB algorithm, the model learns which dishes
        generate the highest served-to-planned efficiency and recommends items that help:

        - Reduce food waste across schools  
        - Improve production planning accuracy  
        - Support nutrition staff with daily data-driven decisions  

        The project uses real production data from Fairfax County Public Schools (FCPS),
        learning demand patterns for each school, date, and meal type. The final system
        generates recommendations and visual analytics that can be used operationally to
        improve district-wide meal planning efficiency.

## Objective
        The goal is to create a CMAB-based tool that helps:

        - Cafeteria managers  
        - School nutritionists  
        - District-level administrators  
        - Researchers  

        The system optimizes production planning by:

        - Learning demand patterns based on contextual features  
        - Identifying dishes with high efficiency (served-to-planned ratio)  
        - Minimizing overproduction and waste  
        - Supporting scalable district-wide decision-making  

## Dataset
        The dataset consists of FCPS meal production records. Each entry includes:

        - School  
        - Date  
        - Meal type  
        - Dish name  
        - Planned servings  
        - Actual servings  
        - Production cost  
        - Waste (discarded + leftover cost)  

        Dataset summary:

        - 7,940 contextual rounds  
        - 321 unique dishes (arms)  
        - Data across ~181 schools and multiple meal periods  

## Rationale
        FCPS faces a recurring challenge:

        - Overproduction leads to high financial waste  
        - Low-demand dishes increase leftover cost  
        - School-level demand patterns vary significantly  
        - Staff lack automated analytics tools  

        A contextual bandit approach enables:

        - Modeling school-specific demand preferences  
        - Choosing dishes that maximize serving efficiency  
        - Providing recommendations without requiring ML expertise  
        - Clear interpretability for non-technical users  

        This project combines reinforcement learning with real school operations
        to produce a practical waste-reduction solution.

## System Architecture & Methodology
        The pipeline contains five major components:

1. Environment (utils/env.py)
        Builds contextual rounds from FCPS data.

        - load_data() → Load and clean production dataset  
        - get_states() → Construct contextual feature matrices  
        - get_actions() → Identify valid dishes per time step  
        - get_health_scores() → (Optional) nutritional scoring  

2. Bandit Model (models.py)
        Implements LinUCB and Random Baseline.

        - action() → Select arm with highest UCB score  
        - train() → Offline replay training  
        - calculate_reward() → Served / Planned efficiency  
        - update() → Update A and b matrices  
        - reset() → Reset learning state  
        - recommend() → Predict top-k dishes  

3. Driver Script (main.py)
        Coordinates the full training flow.

        - Loads dataset  
        - Iterates through all contextual rounds  
        - Trains LinUCB and baseline  
        - Saves recommendations and metrics  

4. Evaluation Tools (visualizations.py, utils/metrics.py)
        Metrics:

        - Cumulative reward  
        - Cumulative regret  
        - Exploration ratio  
        - Rolling average reward  
        - Selection frequency  

        Visuals include:

        - Reward & regret curves  
        - Uncertainty decay  
        - Rolling reward  
        - Top-selected dishes  
        - School-level prediction distributions   

## Contextual Bandit Formulation
        At each round t, the context includes:

        - School (one-hot)  
        - Meal type (one-hot)  
        - Date (day, month)  
        - Planned and served counts  
        - Production & waste cost features  

        Arms:
        - All 321 dishes, masked by daily availability  

## Reward

Default reward:

```python
if Planned_Total == 0:
    reward = 0
else:
    reward = Served_Total / Planned_Total

Parameter updates:

A_a ← A_a + x xᵀ
b_a ← b_a + r x
θ_a = A_a⁻¹ b_a

## Evaluation Protocol
        Offline replay evaluation:

        - Observe historical context  
        - Choose an arm  
        - Look up historical reward  
        - Compute regret vs. the optimal arm for that round  
        - Track cumulative metrics  

        Baseline:
        - Random Policy (uniform random among available dishes)

        Metrics:
        - Cumulative reward  
        - Cumulative regret  
        - Exploration ratio  
        - Uncertainty  
        - Arm frequencies  

## Results
### Summary Table

| Metric              | LinUCB | Random Baseline |
|---------------------|--------|-----------------|
| Total rounds        | 7,940  | 7,940           |
| Unique dishes       | 321    | 321             |
| Cumulative reward   | 369.57 | 90.44           |
| Cumulative regret   | 574.18 | —               |
| Exploration ratio   | 1.69%  | 100%            |

## Key takeaway:
LinUCB achieved over 4× higher cumulative reward compared to the random baseline.

Major Findings
        - Rapid convergence: uncertainty → 0 early  
        - A small set of high-performing dishes dominates  
        - Chickpeas, Fat-Free Milk, Italian Dressing rank consistently top  
        - Rolling average reward stabilizes over time  
        - School-level patterns reveal meaningful demand differences   

## How to Run

Full instructions in **[RUN.md](RUN.md)**

Typical workflow:

python main.py          # Train & evaluate LinUCB
python benchmark.py     # Run benchmarking trials

## Project Timeline
| Component                | Duration |
|--------------------------|----------|
| RL Framework             | 1 week   |
| Environment (env.py)     | 2 weeks  |
| Model (model.py, main.py)| 5 weeks  |
| Metrics & Plots          | 2 weeks  |
| Benchmarking             | 2 weeks  |
| Research Paper Writing   | Parallel |


## Advisor:
Prof. Amir Jafari
Email: ajafari@gwu.edu

## Contributor:
Varun Gholap - varun.gholap@gwu.edu
Neeraj Magdum - neerajshashikant.magadum@gwu.edu