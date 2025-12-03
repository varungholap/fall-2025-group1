# Capstone Proposal
## Health-Aware School Meal Recommendations with Contextual Bandits
### Authors: Neeraj Shashikant Magadum, Varun Gholap
### Advisor: Prof. Amir Jafari
#### The George Washington University, Washington DC
#### M.S. in Data Science – Fall 2025
## Overview
        This project introduces an adaptive recommendation system for K–12 school meal 
        planning using Contextual Multi-Armed Bandits (CMAB). The system helps school 
        nutrition staff optimize three competing objectives:

        - Improve student health by recommending nutritious meals  
        - Increase participation through appealing food choices  
        - Reduce food waste by aligning production with estimated demand  

        Using historical production data from Fairfax County Public Schools (FCPS), the 
        system learns student preferences in context (school, date, meal type) and 
        recommends meals that balance popularity and nutritional value. The LinUCB 
        algorithm is used to dynamically update recommendations while balancing 
        exploration and exploitation.

        The project aims to develop an open-source tool that empowers non-technical school 
        nutrition professionals and researchers to make data-driven decisions for healthier, 
        more sustainable school meal programs.

## Objective
        The goal of this project is to create a free and open-source CMAB-based 
        recommendation tool usable by:

        - School nutritionists  
        - Cafeteria managers  
        - District administrators  
        - Researchers  

        The system recommends meals that optimize health, participation, and waste 
        reduction by:

        - Learning student preferences across contexts  
        - Balancing exploration vs. exploitation  
        - Encouraging nutrient-dense meal options  
        - Forecasting demand more accurately  

        The broader goal is to provide an accessible research and operational tool for 
        data-driven school nutrition planning.

## Dataset
        The dataset consists of production and sales records from Fairfax County 
        Public Schools (FCPS). Each row includes:

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
        - 321 unique menu items (arms)  
        - Data spans multiple schools and meal services  

## Rationale
        School nutrition teams face a difficult challenge:

        - Healthy meals may be unpopular  
        - Popular meals may not meet nutrition standards  
        - Overproduction leads to food waste and financial loss  
        - Staff often lack advanced analytics tools  

        A contextual bandit approach enables:

        - Learning preferences for each school, date, and meal context  
        - Safe exploration of healthier alternatives  
        - Improved demand forecasting  
        - Interpretable recommendations  

        This project bridges public health nutrition with reinforcement learning, 
        producing a practical analytics tool for real-world school meal planning.

## System Architecture & Methodology
        The system pipeline consists of five major components:

1. Environment (utils/env.py)
        Constructs contextual rounds from FCPS data.

        - load_data() → Load cleaned CSV  
        - get_states() → Create feature matrices  
        - get_actions() → Define valid arms per time step  
        - get_health_scores() → Provide nutrition scores for dishes  

2. Bandit Model (model.py)
        Implements the LinUCB contextual bandit algorithm.

        - action() → Select optimal arm  
        - train() → Run offline replay  
        - calculate_reward() → Compute served-to-planned ratio (+ health factor)  
        - update() → Ridge regression update  
        - reset() → Clear model parameters  
        - save() → Save learned model  
        - recommend() → Generate recommendations  

3. Driver Script (main.py)
        Coordinates the environment and the model:

        - Loads dataset  
        - Iterates through contextual rounds  
        - Trains LinUCB  
        - Evaluates and logs performance  

4. Evaluation Tools (utils/metrics.py, utils/plot.py)
        Metrics:

        - Cumulative reward  
        - Cumulative regret  
        - Exploration ratio  
        - Rolling averages  
        - Arm selection frequency  
        - Rank distribution  

        Visualizations include:

        - Reward & regret curves  
        - Uncertainty decay  
        - Rolling reward  
        - Top recommended dishes  
        - School-level score distributions  

5. Benchmarking (benchmark.py)
        Provides systematic experimentation:

        - 10+ repeated trials  
        - Comparison across α values (exploration)  
        - Sweeps over λ (health weighting)  
        - Saves all results to CSV  

Contextual Bandit Formulation
        At each round t, the model receives a "context":

        - One-hot encoded school  
        - Meal type  
        - Date features (day, month)  
        - Planned & served quantities  
        - Production and waste cost metrics  

        Arms:

        - All 321 menu items, masked by availability  

## Reward

Default reward:

if Planned_Total == 0:
    reward = 0
else:
    reward = Served_Total / Planned_Total


Health-aware reward:

reward = (1 - λ) * (Served / Planned) + λ * Health_Score

LinUCB Learning Algorithm

Predicted reward for arm a:

score = θᵀx + α * sqrt(xᵀA⁻¹x)


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


## Plots (see /utils/plot.py):

        - Cumulative reward & regret  
        - Uncertainty decay  
        - Rolling reward  
        - Arm-selection frequency  
        - Rank distribution per dish  
        - School-wise prediction distributions  

## Discussion
        The contextual bandit approach works well even with:

        - Sparse data  
        - Noise in planning and serving metrics  
        - Strong variability across dates and schools  

## Strengths
        - Major performance improvement over baseline  
        - Clear convergence behavior  
        - Transparent, interpretable recommendations  
        - Actionable insights for school nutrition planning  

## Challenges
        - Reward noise due to attendance/weather  
        - Non-stationary student preferences  
        - Rare dishes remain poorly estimated  
        - Current reward lacks multi-objective considerations  

## Future Extensions
        - Multi-objective reward functions  
        - Neural bandits / Thompson sampling  
        - Sliding-window or decayed learning  
        - Multi-day planning using full RL  
        - Integrating menu budgets and USDA nutrition compliance  

## Conclusion
        This project demonstrates that LinUCB contextual bandits significantly 
        outperform uninformed strategies for K–12 school meal recommendations.


## Key conclusions:

        - 4× higher cumulative reward than random  
        - Stable, interpretable recommendations  
        - Insightful modeling of student preference patterns  
        - Strong potential for real-world waste reduction  
        - A practical foundation for future decision-support tools  

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
Varun Gholap – varung@gwu.edu