import pandas as pd
import matplotlib.pyplot as plt

# load rankings data here:
wood = pd.read_csv('Golden_Ticket_Award_Winners_Wood.csv')
steel = pd.read_csv('Golden_Ticket_Award_Winners_Steel.csv')
print(wood.head())
print(steel.head())
print('Steel Rollercoasters: ' + str(len(steel)))
print('Wood Rollercoasters: ' + str(len(wood)))

print('Steel Suppliers: ' +str(steel.Supplier.nunique()))
print('Wood Suppliers: ' +str(wood.Supplier.nunique()))

steel_counts = steel.groupby('Year of Rank').Name.count().reset_index()
wood_counts = wood.groupby('Year of Rank').Name.count().reset_index()
print('Steel Ranks by Year')
print(steel_counts)
print('Wood Ranks by Year')
print(wood_counts)

# write function to plot rankings over time for 1 roller coaster here:
def plot_roller_coaster(name, rankings_df, park_name):
    coaster_rankings = rankings_df[(rankings_df.Park == park_name) & (rankings_df.Name == name)]
    ax = plt.subplot()
    plt.title('Roller Coaster Ranking per Year')
    plt.plot(coaster_rankings['Year of Rank'], coaster_rankings.Rank, color = 'blue', marker = 'o' )
    plt.xlabel('Years')
    plt.ylabel('Ranking')
    ax.invert_yaxis()
    plt.show()
   
plt.clf()
plot_roller_coaster('El Toro', wood, 'Six Flags Great Adventure')

# write function to plot rankings over time for 2 roller coasters here:
def plot_2_roller_coaster(name1, name2, rankings_df, park_name1, park_name2):
    coaster1_rankings = rankings_df[(rankings_df.Park == park_name1) & (rankings_df.Name == name1)]
    coaster2_rankings = rankings_df[(rankings_df.Park == park_name2) & (rankings_df.Name == name2)]
    ax = plt.subplot()
    plt.title(name1 + ' vs '+ name2 + ' Rankings')
    plt.plot(coaster1_rankings['Year of Rank'], coaster1_rankings.Rank, color = 'blue', marker = 'o' )
    plt.plot(coaster2_rankings['Year of Rank'], coaster2_rankings.Rank, color = 'red', marker = 'v' )
    plt.xlabel('Years')
    plt.ylabel('Ranking')
    ax.invert_yaxis()
    plt.legend(['EL Toro', 'Boulder Dash'])
    plt.show()

plt.clf()
plot_2_roller_coaster('El Toro', 'Boulder Dash', wood, 'Six Flags Great Adventure', 'Lake Compounce')

# write function to plot top n rankings over time here:
def plot_n_ranked_roller_coaster(n, rankings_df):
    top_n_rankings = rankings_df[rankings_df.Rank <= n]
    ax = plt.subplot()
    plt.title('Top ' + str(n) + 'Ranked Roller Coasters per Year')
    coasters = set(top_n_rankings['Name'])
    labels = []
    for coaster in coasters:
        coaster_rankings = top_n_rankings[(top_n_rankings.Name == coaster)]
        plt.plot(coaster_rankings['Year of Rank'], coaster_rankings.Rank)
        labels.append(coaster)
    plt.xlabel('Years')
    plt.ylabel('Ranking')
    ax.invert_yaxis()
    plt.legend(labels)
    plt.show()
plt.clf()
plot_n_ranked_roller_coaster(5, wood)

# load roller coaster data here:
roller_coasters = pd.read_csv('roller_coasters.csv')
roller_coasters = roller_coasters.dropna()
print(roller_coasters.head())


# write function to plot histogram of column values here:
def plot_hist_coaster(df, name):
    if (name == 'height'):
        column_to_plot = df.height[df['height'] <= 140]
    else:
        column_to_plot = df[name]
    plt.title('Distribution of Roller Coasters by '+ name)
    plt.xlabel(name)
    plt.ylabel('Number of Roller Coasters')
    plt.hist(column_to_plot)
    plt.show()
    plt.clf()
plt.clf()
plot_hist_coaster(roller_coasters, 'height')
# write function to plot inversions by coaster at a park here:
def plot_bar_inversions(df,park):
    coasters_df = df[df.park == park]
    coasters_df = coasters_df.sort_values('num_inversions', ascending=False)
    plt.bar(coasters_df.name, coasters_df.num_inversions)
    plt.title('Number of inverts per Roller Coaster in park '+ park)
    plt.xlabel('Roller Coasters')
    plt.ylabel('Inverts')
    plt.xticks(rotation = 60)
    plt.show()
    plt.clf()
plt.clf()
plot_bar_inversions(roller_coasters, 'Walygator Parc')

# write function to plot pie chart of operating status here:
def plot_pie_operating(df):
    operating = df[df.status == 'status.operating']
    closed = df[df.status == 'status.closed.definitely']
    status_counts = [len(operating) , len(closed)] 
    plt.title('Pct Operating vs Closed Roller Coasters')
    plt.pie(status_counts, labels=['Operating', 'Closed'],autopct= '%d%%')
    plt.axis('equal')
    plt.show()
    plt.clf()
   
plt.clf()
plot_pie_operating(roller_coasters)
# write function to create scatter plot of any two numeric columns here:
plt.clf()

# 9
# Create a function to plot scatter of any two columns
def scatter_coasters(df, col1, col2):
    #Creating the 2D scatter graph, x against y
    plt.title('Visualization of ' + col1 + ' vs ' + col2)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.scatter(df[col1], df[col2], color='red', marker='o')
    plt.show()
plt.clf()
scatter_coasters(roller_coasters, 'speed', 'length')


# write function to plot pie chart of seating type here:
def plot_bar_seating_type(df):
    ax = plt.subplot(1,1,1)
    plt.title('Quantity of Seating type Roller Coasters')
    plt.bar( range(len(df.seating_type)), df.park)
    ax.set_xticks(range(len(df.seating_type)))
    ax.set_xticklabels(df.seating_type, rotation= 45)
    plt.xlabel('Seating Types')
    plt.ylabel('Number')
    plt.show()
    plt.clf()
   
plt.clf()
seating_types = roller_coasters.groupby('seating_type').park.count().reset_index()
seating_types = seating_types.sort_values('park', ascending=False)
#set_seating_types = set(roller_coasters.seating_type)
#print(set_seating_types)
plot_bar_seating_type(seating_types)

#print(roller_coasters.seating_type.nunique())
#seating_types = roller_coasters.groupby('seating_type').park.count().reset_index()
#print(seating_types.park)

# Calculate average numeric field for roller coasters by seating type
def calc_data_seating_type(df, col1, col2):
    seating_types_data = df.groupby(col1)[col2].mean().reset_index()
    seating_types_data = seating_types_data.sort_values(col2, ascending=False)
    ax = plt.subplot(1,1,1)
    plt.title('Average ' + col2 + ' by Seating type')
    plt.bar( range(len(seating_types_data[col1])), seating_types_data[col2])
    ax.set_xticks(range(len(seating_types_data[col1])))
    ax.set_xticklabels(seating_types_data[col1], rotation= 45)
    plt.xlabel('Seating Types')
    plt.ylabel(col2)
    plt.show()
    plt.clf()

calc_data_seating_type(roller_coasters, 'seating_type', 'height')
calc_data_seating_type(roller_coasters, 'seating_type', 'length')
calc_data_seating_type(roller_coasters, 'seating_type', 'speed')
calc_data_seating_type(roller_coasters, 'seating_type', 'num_inversions')



# write function to plot a bar chart of the most seating types made by manufacturers here:
def plot_bar_park_seating_type(df):
    ax = plt.subplot(1,1,1)
    plt.title('Quantity of Manufacturers by Seating type Roller Coasters')
    plt.bar( range(len(df.seating_type)), df.manufacturer)
    ax.set_xticks(range(len(df.seating_type)))
    ax.set_xticklabels(df.seating_type, rotation= 45)
    plt.xlabel('Seating Types')
    plt.ylabel('Qty of Manufacterers')
    plt.show()
    plt.clf()    
# Logic to get the Manufacturer's seating type specialty
manufacturer_data = roller_coasters.groupby(['manufacturer', 'seating_type']).park.count().reset_index()
data = []
set_manufacturer = set(manufacturer_data.manufacturer)
# Calculate max number of seating_type roller coasters by Manufacturer
for manufacturer in set_manufacturer:
    seating_type_data = manufacturer_data[manufacturer_data.manufacturer == manufacturer]
    # Pick the value with the highest count of the seating_types
    max_value = seating_type_data.park.max()
    # Find the seating_type that belongs to the max count
    specialty = seating_type_data[seating_type_data.park == max_value].seating_type.values
    # Create a list of seating _type with  the most counts
    data.append({'manufacturer':manufacturer, 'seating_type': specialty[0], 'specialty': max_value})
# Build a data frame with the manufacturer's most seating_types count.    
specialty_df = pd.DataFrame(data)
# Count number of manufacturer with their seating_type specialty (Most counts of seating_types)
seating_type_count = specialty_df.groupby(['seating_type']).manufacturer.count().reset_index()
seating_type_count = seating_type_count.sort_values('manufacturer', ascending=False)

plot_bar_park_seating_type(seating_type_count)

# Calculate manufacturer data like height, length, speed, seting type
def plot_manufacturer_data(df, name):
    park_data = df[df.manufacturer == name]
    avg_height = park_data.height.mean()
    avg_speed = park_data.speed.mean()
    avg_length = park_data.length.mean()
    avg_inversions = park_data.num_inversions.mean() * 10 # Help make then stand out in the bar
    plt.title('Average data for Manufacturer ' + name)
    ax = plt.subplot()
    plt.bar(range(4), [avg_height, avg_speed, avg_length, avg_inversions])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Avg Height(m)', 'Avg Speed (kmh)', 'Avg Length (km)', 'Avg # of Inversions X 10'], rotation=90)
    plt.show()
    plt.clf()

plot_manufacturer_data(roller_coasters, 'Vekoma')
plot_manufacturer_data(roller_coasters, 'Zamperla')
plot_manufacturer_data(roller_coasters, 'William J. Cobb')
plot_manufacturer_data(roller_coasters, 'Soquet')    

# Calculate manufacturer data like height, length, speed, seting type
def plot_park_data(df, name):
    park_data = df[df.park == name]
    avg_height = park_data.height.mean()
    avg_speed = park_data.speed.mean()
    avg_length = park_data.length.mean()
    avg_inversions = park_data.num_inversions.mean() * 10 # Help make then stand out in the bar
    plt.title('Average data for Park ' + name)
    ax = plt.subplot()
    plt.bar(range(4), [avg_height, avg_speed, avg_length, avg_inversions])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Avg Height(m)', 'Avg Speed (kmh)', 'Avg Length (km)', 'Avg # of Inversions X 10'], rotation=90)
    plt.show()
    plt.clf()

plot_park_data(roller_coasters, 'Parc Asterix')
plot_park_data(roller_coasters, 'Bobbejaanland')
plot_park_data(roller_coasters, 'Terra Mítica')
plot_park_data(roller_coasters, 'Walygator Parc')