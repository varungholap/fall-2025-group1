# How to Run

## 1. Install Dependencies
To run the project, you need to install the required Python libraries. You can do this using pip and the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 2. Run the Main Script
The main script for this project is `src/main.py`. To run it, navigate to the `src` directory and execute the script.

```bash
python src/main.py
```

This will run the LinUCB bandit algorithm on the dataset and generate recommendations and plots.

## 3. Where to Check the Results
After running the main script, the following outputs will be generated in the `reports` directory:

- **`reports/recommendations.csv`**: This file contains the top recommended dishes for each school and meal type.
- **`reports/plots/`**: This directory contains various plots that visualize the performance of the LinUCB algorithm and the characteristics of the recommendations.
