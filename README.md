# Mercari-Price-Suggestion

Introduction:
It can be hard to know how much something’s really worth. Small details can mean big differences in pricing. Product pricing gets even harder at scale, considering just how many products are sold online. Clothing has strong seasonal pricing trends and is heavily influenced by brand names, while electronics have fluctuating prices based on product specs.
Mercari, Japan’s biggest community-powered shopping app, knows this problem deeply. They’d like to offer pricing suggestions to sellers, but this is tough because their sellers are enabled to put just about anything, or any bundle of things, on Mercari's marketplace.

Objective:
Build an algorithm that automatically suggests the right product prices.

Data Description:
Different attributes present in the data are as follows:
1) train_id or test_id - the id of the listing
2) name - the title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]
3) item_condition_id - the condition of the items provided by the seller
4) category_name - category of the listing
5) brand_name
6) price - the price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict.
7) shipping - 1 if shipping fee is paid by seller and 0 by buyer
8) item_description - the full description of the item.
