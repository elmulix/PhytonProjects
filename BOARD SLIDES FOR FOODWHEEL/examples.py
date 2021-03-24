from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

restaurants =  pd.read_csv('restaurants.csv')
print(restaurants.head())
cuisine_options_count = restaurants.cuisine.nunique()
cuisine_counts = restaurants.groupby('cuisine').name.count().reset_index()
print(cuisine_counts)

###############################################
# What cuisines does FoodWheel offer?
import codecademylib3
from matplotlib import pyplot as plt
import pandas as pd

restaurants = pd.read_csv('restaurants.csv')

cuisine_counts = restaurants.groupby('cuisine')\
                            .name.count()\
                            .reset_index()
                            
cuisines = cuisine_counts.cuisine.values
counts = cuisine_counts.name.values

plt.title('Pct of restaurants per cuisine')
plt.pie(counts, labels= cuisines, autopct= '%d%%')
plt.axis('equal')
plt.show()

###########################################
# Orders Over Time
import codecademylib
from matplotlib import pyplot as plt
import pandas as pd

orders = pd.read_csv('orders.csv')

orders['month'] = orders.date.apply(lambda x: x.split('-')[0])

avg_order = orders.groupby('month').price.mean().reset_index()

std_order = orders.groupby('month').price.std().reset_index()

# Make a graph
ax = plt.subplot(1,1,1)
bar_heights = avg_order.price
bar_errors = std_order.price
plt.bar(range(len(bar_heights)), bar_heights, yerr=bar_errors, capsize=5)
ax.set_xticks(range(len(bar_heights)))
ax.set_xticklabels(['April','May','June','July','August','September','October','November','December'])
plt.title('Average order per month')
plt.ylabel('Amount')
plt.show()

######################################
# Customer Types
import codecademylib
from matplotlib import pyplot as plt
import pandas as pd

orders = pd.read_csv('orders.csv')
customer_amount = orders.groupby('customer_id').price.sum().reset_index()
print(customer_amount.head())
plt.title('Amount spent per customer')
plt.xlabel('Amount')
plt.ylabel('Number of customers')
plt.hist(customer_amount.price.values, range=(0,200), bins=40)
plt.show()
