Health-Aware School Meal Recommendations with Contextual Bandits
Capstone Project – The George Washington University

Authors: Neeraj Shashikant Magadum, Varun Gholap
Advisor: Prof. Amir Jafari
Program: M.S. in Data Science
Semester: Fall 2025

Overview

This project introduces an adaptive recommendation system for K–12 school meal planning using Contextual Multi-Armed Bandits (CMAB). The system helps school nutrition staff optimize three competing objectives:

Improve student health by recommending nutritious meals

Increase participation through appealing food choices

Reduce food waste by aligning production with estimated demand

Using historical production data from Fairfax County Public Schools (FCPS), our system learns student preferences in context (school, date, meal type) and recommends meals that balance popularity and nutritional value. The underlying algorithm, LinUCB, adaptively updates its recommendations using reinforcement learning principles.

This project is being developed as an open-source tool to empower non-technical school nutritionists and researchers to make data-informed decisions.

Table of Contents

Objective

Dataset

Rationale

System Architecture & Methodology

Contextual Bandit Formulation

LinUCB Learning Algorithm

Evaluation Protocol

Results

Discussion

Conclusion

How to Run

Project Timeline

Contact

Objective

The goal of this project is to create a free and open-source CMAB-based recommendation tool that can be used by:

School nutritionists

Cafeteria managers

District administrators

Researchers

The system recommends meals that optimize student health, participation, and waste reduction by:

Learning student preferences across contexts

Balancing exploration vs. exploitation

Encouraging nutrient-dense meal options

Forecasting demand more accurately

The broader aim is to build an accessible research tool capable of supporting data-driven school nutrition planning.

Dataset

We use production and sales data provided by Fairfax County Public Schools (FCPS).

Each row includes:

School

Date

Meal type

Dish name

Planned servings

Actual servings

Production cost

Waste (discarded + leftover cost)

Over the full dataset:

7,940 contextual rounds

321 unique dishes (arms)

Data spans multiple schools and meal services

Rationale

School nutrition teams face a difficult challenge:

Healthy meals may be unpopular

Popular meals may not meet nutrition goals

Overproduction leads to food waste and financial loss

Staff usually lack advanced analytics tools

A contextual bandit approach allows dynamic, real-time decision-making:

Learns which meals perform well in each school context

Encourages exploration of healthier alternatives

Predicts demand to minimize waste

Provides interpretable recommendations for non-technical users

This project bridges public health nutrition and reinforcement learning, producing a practical tool for school operations.

System Architecture & Methodology

The pipeline consists of several modules:

1. Environment (utils/env.py)

Construct contextual rounds from FCPS data.

load_data() – load cleaned CSV

get_states() – feature matrices for each round

get_actions() – available meals at each time step

get_health_scores() – nutrition-based scoring per meal

2. Bandit Model (model.py)

Implements the full LinUCB algorithm.

action() – choose optimal arm

train() – iteratively update parameters

calculate_reward() – served-to-planned ratio (+ health factor)

update() – ridge regression update

reset() – clear parameters

save() – persist model

recommend() – generate recommendations

3. Driver Script (main.py)

Coordinates environment + model for full training loop.

4. Evaluation Tools (utils/metrics.py / utils/plot.py)

Cumulative reward

Cumulative regret

Moving averages

Exploration metrics

Arm-selection frequencies

Rank distribution plots

5. Benchmarking (benchmark.py)

Runs 10+ trials

Compares multiple α (exploration) values

Sweeps λ (health weighting) values

Saves results to CSV

Contextual Bandit Formulation

At each round t, the model receives:

Context

Feature vector combining:

One-hot encoded school

Meal type

Day of week, month

Planned & served quantities

Production and waste costs

Arms

All 321 unique menu items, masked by daily availability.

Reward

The default reward:

if Planned_Total == 0:
    reward = 0
else:
    reward = Served_Total / Planned_Total


Extended health-aware reward:

reward = (1 - λ) * (Served / Planned) + λ * Health_Score

LinUCB Learning Algorithm

The predicted reward for arm a:

score = θᵀx + α * sqrt(xᵀA⁻¹x)


Where:

θ – learned parameter weights

A – design matrix (updated per arm)

α – exploration parameter

After observing reward r, LinUCB updates:

A_a ← A_a + x xᵀ
b_a ← b_a + r x
θ_a = A_a⁻¹ b_a

Evaluation Protocol

Evaluation uses offline replay, where:

Model observes historical context

Chooses an arm

Receives reward from actual historical data

Computes regret vs. optimal arm that day

Metrics:

Cumulative reward

Cumulative regret

Exploration ratio

Uncertainty decay

Selection frequency

Baseline: Random Policy (uniform random valid arm).

Results
Summary Table
Metric	LinUCB	Random Baseline
Total rounds	7,940	7,940
Unique dishes	321	321
Cumulative reward	369.57	90.44
Cumulative regret	574.18	—
Exploration ratio	1.69%	—

Key takeaway:
LinUCB achieved more than 4× the cumulative reward of the random policy.

Major findings

LinUCB rapidly converges (uncertainty → 0 early)

A small set of high-performing dishes dominate recommendations

Chickpeas, Fat-Free Milk, Italian Dressing were consistently top-ranked

Reward moving average stabilizes over time

School-level patterns indicate real differences in demand

All plots are implemented in /utils/plot.py:

Cumulative reward & regret

Uncertainty vs. rounds

Rolling average reward

Arm-selection frequencies

Rank distribution for specific dishes

School-wise score distributions

Discussion

Our study demonstrates that a contextual bandit is a powerful tool for school meal optimization, even in a real-world dataset that is:

Sparse

Noisy

Highly variable across time

Strengths

Major reward improvement over baseline

Clear convergence behavior

Interpretable arm preferences

Provides actionable insights to administrators

Challenges

Reward signal influenced by external factors (attendance, weather, events)

Non-stationary student preferences

Many items appear too rarely to learn strong estimates

Current reward does not include cost/variety/nutritional constraints explicitly

Future improvements

Multi-objective reward functions

Thompson Sampling, Neural Bandits

Sliding-window or discounting for non-stationarity

Multi-day planning via full RL environment

Integration with menu management dashboards

Factoring nutrition, budget, and USDA compliance into policy

Conclusion

This project successfully demonstrates that LinUCB contextual bandits significantly outperform uninformed strategies in recommending K–12 school meals.

Key conclusions:

LinUCB achieved 4× higher cumulative reward than random

Learned stable, interpretable recommendations

Provided deep insight into student food preference patterns

Can help reduce waste and improve participation

Can form the foundation of a practical decision-support tool

Bandit-based decision systems hold real promise for school nutrition optimization, and this work provides a strong foundation for continued development.

How to Run

Full instructions are provided in:
RUN.md

Typical workflow:

python main.py          # Train & evaluate LinUCB
python benchmark.py     # Run benchmarking trials

Project Timeline
Component	Duration
RL Framework	1 week
Environment (env.py)	2 weeks
Model (model.py, main.py)	5 weeks
Metrics & Plots	2 weeks
Benchmarking	2 weeks
Research Paper Writing	Parallel to metrics/plots
Contact

Advisor:
Prof. Amir Jafari
Email: ajafari@gwu.edu

Contributor:
Varun Gholap – varung@gwu.edu