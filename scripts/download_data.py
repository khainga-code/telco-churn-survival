"""Download the IBM Telco Customer Churn dataset into data/."""
from telco_churn.data import download

if __name__ == "__main__":
    print(f"Saved to {download()}")
