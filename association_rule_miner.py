import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

class AssociationRuleMiner:
    def __init__(self, file_path, min_support=0.005, confidence_threshold=0.2, sep=';'):
        """
        Initialize the AssociationRuleMiner with file path, minimum support, and confidence threshold.

        :param file_path: str - Path to the CSV file containing transaction data.
        :param min_support: float - Minimum support for the apriori algorithm.
        :param confidence_threshold: float - Confidence threshold for filtering the association rules.
        :param sep: str - Separator used in the CSV file (default is ';').
        """
        self.file_path = file_path
        self.min_support = min_support
        self.confidence_threshold = confidence_threshold
        self.sep = sep
        self.rules = None

    def load_data(self):
        """
        Load the transaction data from a CSV file.

        :return: DataFrame containing transactions grouped by 'Payment ID'.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        data = pd.read_csv(self.file_path, sep=self.sep)
        transactions = data.groupby('Payment ID')['Item'].agg(lambda x: list(x)).reset_index()['Item']
        return transactions

    def apply_one_hot_encoding(self, transactions):
        """
        Apply one-hot encoding to the transaction data.

        :param transactions: Series - Series of lists containing the items in each transaction.
        :return: DataFrame - One-hot encoded DataFrame.
        """
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_array, columns=te.columns_)

    def generate_frequent_itemsets(self, df):
        """
        Generate frequent itemsets using the Apriori algorithm.

        :param df: DataFrame - One-hot encoded transaction data.
        :return: DataFrame - Frequent itemsets.
        """
        return apriori(df, min_support=self.min_support, use_colnames=True)

    def generate_association_rules(self):
        """
        Generate association rules based on the frequent itemsets and confidence threshold.

        :return: DataFrame - Association rules.
        """
        transactions = self.load_data()
        one_hot_df = self.apply_one_hot_encoding(transactions)
        frequent_itemsets = self.generate_frequent_itemsets(one_hot_df)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=self.confidence_threshold)
        self.rules = rules[rules['confidence'] >= self.confidence_threshold]
        return self.rules

    def save_rules_to_csv(self, output_path):
        """
        Save the generated association rules to a CSV file.

        :param output_path: str - Path where the rules should be saved.
        """
        if self.rules is None:
            raise ValueError("No rules have been generated. Please run generate_association_rules() first.")
        
        self.rules.to_csv(output_path, index=False)
        print(f"Association rules saved to {output_path}")

# Example usage
if __name__ == "__main__":
    file_path = 'file-path.csv'
    output_file = 'output.csv'
    
    miner = AssociationRuleMiner(file_path, min_support=0.005, confidence_threshold=0.2)
    rules = miner.generate_association_rules()
    print(rules)
    
    miner.save_rules_to_csv(output_file)