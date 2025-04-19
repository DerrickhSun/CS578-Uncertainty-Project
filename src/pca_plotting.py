import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris # Example dataset
import pandas as pd
import numpy as np

# Load data (replace with your data)
iris = pd.read_csv("testing.csv")

features = ["item_id", "wday", "month", "year", "snap_CA", "sell_price", "event_name_1_filled", "event_type_1_filled", "sell_price_filled", 
            "dept_id:HOBBIES_1", "store_id:CA_1","store_id:CA_2","store_id:CA_3","store_id:CA_4","cat_id:HOBBIES", "state_id:CA", "event_name_1:Chanukah End", "event_name_1:Christmas", 
            "event_name_1:Cinco De Mayo", "event_name_1:ColumbusDay", "event_name_1:Easter", "event_name_1:Eid al-Fitr", "event_name_1:EidAlAdha", 
            "event_name_1:Father's day", "event_name_1:Halloween", "event_name_1:IndependenceDay", "event_name_1:LaborDay", 
            "event_name_1:LentStart", "event_name_1:LentWeek2", "event_name_1:MartinLutherKingDay", "event_name_1:MemorialDay", 
            "event_name_1:Mother's day", "event_name_1:NBAFinalsEnd", "event_name_1:NBAFinalsStart", "event_name_1:NewYear", 
            "event_name_1:OrthodoxChristmas", "event_name_1:OrthodoxEaster", "event_name_1:Pesach End", "event_name_1:PresidentsDay", 
            "event_name_1:Purim End", "event_name_1:Ramadan starts", "event_name_1:StPatricksDay", "event_name_1:SuperBowl", 
            "event_name_1:Thanksgiving", "event_name_1:ValentinesDay", "event_name_1:VeteransDay", "event_name_1:no_event_name_1", 
            "event_type_1:Cultural", "event_type_1:National", "event_type_1:Religious", "event_type_1:Sporting", 
            "event_type_1:no_event_type_1", "item_store_last_day_sales", "item_store_last_day_sales_filled", 
            "item_store_L7d_day_median_sales_filled", "item_store_L7d_day_median_sales", "item_store_L14d_day_median_sales_filled", 
            "item_store_L14d_day_median_sales", "item_store_L21d_day_median_sales_filled", "item_store_L21d_day_median_sales", 
            "item_store_L28d_day_median_sales_filled", "item_store_L28d_day_median_sales"]
discarded = ["sales", "d", "Unnamed: 0", "Unnamed: 0.1", "id", "state_id:CA", "cat_id:HOBBIES"]
features = [col for col in iris.columns if col not in discarded]

# Jitter function
def jitter(arr, amount=0.2):
    return arr + np.random.uniform(-amount, amount, size=len(arr))

pca = PCA(n_components=2)  # Retain 2 principal components
principal_components = pca.fit_transform(iris[features])
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df = pd.concat([pca_df, np.log(iris[['sales']]+1)], axis=1)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(jitter(pca_df["PC1"]), jitter(pca_df["PC2"]), s=.5, c=pca_df["sales"], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot Colored by log(Sales)')
plt.colorbar(label='log(Sales)')  # Add a color bar to show the mapping
plt.grid(True)
plt.show()