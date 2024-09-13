# Restaurant Sales Recommendation System

This repository contains a Python script that leverages association rule mining to generate insights about frequently purchased items in a restaurant setting. Using the Apriori algorithm, this script helps identify itemsets that are often bought together, which can be used as the foundation for a recommendation system aimed at improving sales.

## Purpose
The main goal of this project is to assist restaurant staff, managers, and sales teams by providing item recommendations based on historical sales data. The system identifies patterns in customer purchases, helping to suggest items that are likely to be bought together. This can optimize upselling strategies and improve customer satisfaction by offering personalized recommendations.

## Features
- **Transaction-based analysis**: Analyzes sales transactions to identify frequently bought-together items.
- **Association rule generation**: Applies the Apriori algorithm to uncover strong association rules based on support and confidence thresholds.
- **Customizable parameters**: Users can adjust the minimum support and confidence levels to suit their dataset.
- **CSV export**: Generated association rules can be saved to a CSV file for further analysis or integration into other systems.

## How to Use
1. Clone the repository to your local machine.
2. Install the required dependencies (listed in `requirements.txt`).
3. Place your sales transaction data in the `data/` folder.
4. Run the script to generate association rules and output the results to a CSV file.
5. Use the generated rules to create a recommendation system for restaurant staff.

## Example
Below is an example of how to run the script:

```bash
python association_rule_miner.py
