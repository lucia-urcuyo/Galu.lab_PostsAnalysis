import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objs as go
from plotly.subplots import make_subplots

ReelsData = pd.read_csv('ReelsData.csv')
PhotosData = pd.read_csv('PhotosData.csv')


#Make a K-means clustering model for likes plays and comments

features = ReelsData[['likes', 'comments', 'plays']]

# Initialize the KMeans model with the desired number of clusters
kmeans = KMeans(n_clusters=3, random_state=420)

# Fit the model to your data
kmeans.fit(features)

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Add the cluster labels to your DataFrame
ReelsData['Cluster'] = cluster_labels


cluster = px.scatter_3d(ReelsData, x='likes', y='comments', z='plays', color='Cluster', opacity=0.7,
                     color_continuous_scale=px.colors.sequential.Viridis)

# Customize the plot appearance
cluster.update_layout(
    scene=dict(
        xaxis_title='Likes',
        yaxis_title='Comments',
        zaxis_title='Plays'
    ),
    title='Clusters'
)

# Show the plot
#cluster.show()

# To choose hte value of k we will mae an elbow plot, above we ran the model with k = 3 but that is based of the following elbow chart
variance_df = pd.DataFrame(columns=['k', 'variance'])

# Iterate through different values of k and calculate the variance for each
# Create lists to store the results
k_values = range(1, 11)  # Adjust the range as needed
variances = []

# Iterate through different values of k and calculate the variance for each
for k in k_values:
    # Extract the relevant features for clustering from the DataFrame
    features = ReelsData[['likes', 'comments', 'plays']]

    # Initialize the KMeans model with the current value of k
    kmeans = KMeans(n_clusters=k, random_state=420)

    # Fit the model to your data
    kmeans.fit(features)

    # Calculate the variance (inertia) and append it to the list
    variance = kmeans.inertia_
    variances.append(variance)

# Create a DataFrame from the lists
variance_df = pd.DataFrame({'k': k_values, 'variance': variances})

# Create a DataFrame from the lists
variance_df = pd.DataFrame({'k': k_values, 'variance': variances})

# Create a plotly figure
elbow = px.line(variance_df, x='k', y='variance', title='Variance vs. Number of Clusters (Elbow Method)')
elbow.update_xaxes(title='Number of Clusters (k)')
elbow.update_yaxes(title='Variance (Inertia)')

# Show the plot
#elbow.show()


#Now we will manipulate the data and create the charts for the Reels data

# count how many reels where posted on each day of the week
postsperday = ReelsData.groupby('day_of_week')['index'].count()

# convert pandas series to dataframes
postdailycount = postsperday.to_frame()

# Set index as new column (day_of_week)
postdailycount.reset_index(inplace=True)

# rename 'index' column to 'dailycount'
postdailycount.rename({'index': 'dailycount'}, axis=1, inplace=True)

# set 'dailycount' column as a variable
dailycount = postdailycount['dailycount']

# get the play counts for each day separately, not using this yet
saturdayposts = postdailycount.loc[postdailycount['day_of_week'] == 'Sat']

# get daily average plays
avgdailyplays = ReelsData.groupby('day_of_week')['plays'].mean()

# convert pandas series to dataframes
dailyplaysmean = avgdailyplays.to_frame()

# Set index as new column (day_of_week)
dailyplaysmean.reset_index(inplace=True)

# add daily count as a new column
dailyplaysmean['dailycount'] = dailycount

# Reorder index according to day of the week
dailyplaysmean = dailyplaysmean.reindex([1, 5, 6, 4, 0, 2, 3])

# Extract data from each day of the week separately
sunday = ReelsData.loc[ReelsData['day_of_week'] == 'Sun']
monday = ReelsData.loc[ReelsData['day_of_week'] == 'Mon']
tuesday = ReelsData.loc[ReelsData['day_of_week'] == 'Tues']
wednesday = ReelsData.loc[ReelsData['day_of_week'] == 'Weds']
thursday = ReelsData.loc[ReelsData['day_of_week'] == 'Thurs']
friday = ReelsData.loc[ReelsData['day_of_week'] == 'Fri']
saturday = ReelsData.loc[ReelsData['day_of_week'] == 'Sat']

# calculate the average of plays per hour each day
meanplayssun = sunday.groupby('hour posted')['plays'].mean()
meanplaysmon = monday.groupby('hour posted')['plays'].mean()
meanplaystues = tuesday.groupby('hour posted')['plays'].mean()
meanplaysweds = wednesday.groupby('hour posted')['plays'].mean()
meanplaysthurs = thursday.groupby('hour posted')['plays'].mean()
meanplaysfri = friday.groupby('hour posted')['plays'].mean()
meanplayssat = saturday.groupby('hour posted')['plays'].mean()

# convert pandas series to dataframes
sunplays = meanplayssun.to_frame()
monplays = meanplaysmon.to_frame()
tuesplays = meanplaystues.to_frame()
wedsplays = meanplaysweds.to_frame()
thursplays = meanplaysthurs.to_frame()
friplays = meanplaysfri.to_frame()
satplays = meanplayssat.to_frame()

# Set index as new column (hour posted), so that it can be used in the barplot
sunplays.reset_index(inplace=True)
monplays.reset_index(inplace=True)
tuesplays.reset_index(inplace=True)
wedsplays.reset_index(inplace=True)
thursplays.reset_index(inplace=True)
friplays.reset_index(inplace=True)
satplays.reset_index(inplace=True)

# calculate the count of plays per hour each day
postcountsun = sunday.groupby('hour posted')['plays'].count()
postcountmon = monday.groupby('hour posted')['plays'].count()
postcounttues = tuesday.groupby('hour posted')['plays'].count()
postcountweds = wednesday.groupby('hour posted')['plays'].count()
postcountthurs = thursday.groupby('hour posted')['plays'].count()
postcountfri = friday.groupby('hour posted')['plays'].count()
postcountsat = saturday.groupby('hour posted')['plays'].count()

# convert pandas series to dataframes
postcsun = postcountsun.to_frame()
postcmon = postcountmon.to_frame()
postctues = postcounttues.to_frame()
postcweds = postcountweds.to_frame()
postcthurs = postcountthurs.to_frame()
postcfri = postcountfri.to_frame()
postcsat = postcountsat.to_frame()

# Rename the column as 'count'
# Set index as new column (hour posted) in order to be able to add it as a new column to the plays charts
# Set the count column as a variable
# Add count column to each day's plays chart
postcsun.rename({'plays': 'count'}, axis=1, inplace=True)
postcsun.reset_index(inplace=True)
count = postcsun['count']
sunplays['count'] = count

postcmon.rename({'plays': 'count'}, axis=1, inplace=True)
postcmon.reset_index(inplace=True)
count = postcmon['count']
monplays['count'] = count

postctues.rename({'plays': 'count'}, axis=1, inplace=True)
postctues.reset_index(inplace=True)
count = postctues['count']
tuesplays['count'] = count

postcweds.rename({'plays': 'count'}, axis=1, inplace=True)
postcweds.reset_index(inplace=True)
count = postcweds['count']
wedsplays['count'] = count

postcthurs.rename({'plays': 'count'}, axis=1, inplace=True)
postcthurs.reset_index(inplace=True)
count = postcthurs['count']
thursplays['count'] = count

postcfri.rename({'plays': 'count'}, axis=1, inplace=True)
postcfri.reset_index(inplace=True)
count = postcfri['count']
friplays['count'] = count

postcsat.rename({'plays': 'count'}, axis=1, inplace=True)
postcsat.reset_index(inplace=True)
count = postcsat['count']
satplays['count'] = count

# Plots need to be created store in a variable and then we can show them in streamlit using st.plotly_chart(chartobject, ..)

# Create bar plots for all dependent variables (y axis the variable, x axis the post number, legend and color: day of week)
likesplot = px.bar(ReelsData, x="index", y="likes", color="day_of_week", title="Likes", hover_data=['shortcap', 'hour posted', 'hashtags'])
# likesplot.show()
enjoyplot = px.bar(ReelsData, x="index", y="enjoyment", color="day_of_week", title="Enjoyment", hover_data=['shortcap', 'hour posted', 'hashtags'])
# enjoyplot.show()
playsplot = px.bar(ReelsData, x="index", y="plays", color="day_of_week", title="Plays", hover_data=['shortcap','hour posted', 'hashtags'])
# playsplot.show()
commentsplot = px.bar(ReelsData, x="index", y="comments", color="day_of_week", title="Comments", hover_data=['shortcap', 'hour posted', 'hashtags'])
# commentsplot.show()

# Make a bar plots of average plays per day
dailyplaysplot = px.bar(dailyplaysmean, x="day_of_week", y="plays", title='Average Daily Plays', hover_data=['dailycount'])
# dailyplaysplot.show()

# Make bar plots of average plays per hour each day
sunplaysplot = px.bar(sunplays, x="hour posted", y="plays", title="Sunday", hover_data=['count'])
# sunplaysplot.show()
monplaysplot = px.bar(monplays, x="hour posted", y="plays", title="Monday", hover_data=['count'])
# monplaysplot.show()
tuesplaysplot = px.bar(tuesplays, x="hour posted", y="plays", title="Tuesday", hover_data=['count'])
# tuesplaysplot.show()
wedsplaysplot = px.bar(wedsplays, x="hour posted", y="plays", title="Wednesday", hover_data=['count'])
# wedsplaysplot.show()
thursplaysplot = px.bar(thursplays, x="hour posted", y="plays", title="Thursday", hover_data=['count'])
# thursplaysplot.show()
friplaysplot = px.bar(friplays, x="hour posted", y="plays", title="Friday", hover_data=['count'])
# friplaysplot.show()
satplaysplot = px.bar(satplays, x="hour posted", y="plays", title='Saturday', hover_data=['count'])
# satplaysplot.show()


histogram_columns = ReelsData.drop(columns=['index','id','permalink','caption','shortcap','Cluster','num_day_of_week'])

histogram_columns = histogram_columns.iloc[:, 1:]
# Create subplots for each numerical variable
# Create subplots for each numerical variable with 2 rows and adjust the grid accordingly
num_rows = 2
num_cols = len(histogram_columns.columns) // num_rows
histograms = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=histogram_columns.columns)

for i, col in enumerate(histogram_columns.columns, 1):
    row = (i - 1) // num_cols + 1  # Calculate the row for the subplot
    col_name = histogram_columns.columns[i - 1]  # Get the column name
    trace = go.Histogram(x=histogram_columns[col_name], name=col_name)  # Use the column name directly
    histograms.add_trace(trace, row=row, col=(i - 1) % num_cols + 1)

# Update subplot layout
histograms.update_layout(
    title='Distribution of Variables',
    showlegend=False,
    height=600,  # Adjust the height as needed
    width=2000,  # Adjust the width as needed
)



# Show the plot
#histograms.show()

cormatrix_columns = histogram_columns.drop(columns=['date','day_of_week'])

# Calculate the correlation matrix
correlation_matrix = cormatrix_columns.corr()

#Create Colors
custom_colors = [[0.0, 'white'], [1.0, 'red']]

# Create a heatmap using Plotly
matrix = px.imshow(
    correlation_matrix,
    labels=dict(x="Features", y="Features", color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    color_continuous_scale=custom_colors,  # You can choose a different color scale
)

# Customize the layout
matrix.update_layout(
    title='Correlation Matrix',
    xaxis=dict(tickangle=-45),
    width=700,  # Adjust the width as needed
    height=700,  # Adjust the height as needed
)

# Add custom annotations
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        matrix.add_annotation(
            x=correlation_matrix.columns[i],
            y=correlation_matrix.columns[j],
            text=str(correlation_matrix.iloc[j, i].round(2)),
            showarrow=False,
            font=dict(color='black')
        )

# Create a title for your Streamlit app
st.title("Galu.lab Instagram Post Analysis: Uncovering Insights and Trends")

# We are going to display the total number of plays, likes, and comments

# Add the total number of plays, likes, and comments from all csv files
# Make them an integer to remove the decimal point
# Make them a string and change the number to thousands
sum1 = int(ReelsData['plays'].sum())
sum1 = str(sum1)
sum1 = (sum1[0:3]+'K')
sum2 = int(ReelsData['likes'].sum() + PhotosData['likes'].sum())
sum2 = str(sum2)
sum2 = (sum2[0:3]+'K')
sum3 = int(ReelsData['comments'].sum() + PhotosData['comments'].sum())
sum3 = str(sum3)
sum3 = (sum3[0:3]+'K')

# Add a title to the numbers created above
title1 = "Plays"
title2 = "Likes"
title3 = "Comments"

# Create a Streamlit column layout
col1, col2, col3 = st.columns(3)

# Place title and sum in the first column with larger text
col1.markdown(f"<h1 style='text-align: center;'>{title1}</h1>", unsafe_allow_html=True)
col1.markdown(f"<h1 style='text-align: center;'>{sum1}</h1>", unsafe_allow_html=True)

# Place title and sum in the second column with larger text
col2.markdown(f"<h1 style='text-align: center;'>{title2}</h1>", unsafe_allow_html=True)
col2.markdown(f"<h1 style='text-align: center;'>{sum2}</h1>", unsafe_allow_html=True)

# Place title and sum in the third column with larger text
col3.markdown(f"<h1 style='text-align: center;'>{title3}</h1>", unsafe_allow_html=True)
col3.markdown(f"<h1 style='text-align: center;'>{sum3}</h1>", unsafe_allow_html=True)


''' '''
''' '''
''' '''
# Subtitle (using Markdown)
st.markdown("<h2>Reels Data Analysis</h2>", unsafe_allow_html=True)

'''The histograms below visually represent data distribution in our Instagram post analysis. They enable us to spot trends in various metrics swiftly and are crucial for identifying patterns, outliers, and areas where our Instagram account's performance can be enhanced.'''
st.plotly_chart(histograms, use_container_width=True)
'''The correlation matrix presented below offers a comprehensive view of the relationships between various factors associated with each Reels post. Through visualizing the correlation coefficients, we gain valuable insights into the strength and direction of connections between these variables. This analysis plays a pivotal role in identifying patterns, dependencies, and associations within our datase.'''
st.plotly_chart(matrix, use_container_width=True)

# Show all charts in streamlit
'''The following charts offer a comprehensive view of our reels' performance over time, measured by key performance indicators (KPIs) such as Comments, Plays, Likes, and Enjoyability (a ratio of likes to plays). These charts provide insights into the evolution of our reel content's performance metrics, helping us track changes, patterns, and shifts in audience engagement. To enhance our analysis, these charts are color-coded based on the 'day of the week,' allowing us to visually assess how different days may impact each KPI. '''
st.plotly_chart(commentsplot, use_container_width=True)
st.plotly_chart(playsplot, use_container_width=True)
st.plotly_chart(likesplot, use_container_width=True)
st.plotly_chart(enjoyplot, use_container_width=True)

'''The following chart offers deeper insights into how the 'day of the week' influences our average plays. It allows us to examine how different days impact the average number of plays our reels receive. This analysis helps us identify trends and patterns related to content engagement throughout the week'''
st.plotly_chart(dailyplaysplot, use_container_width=True)

'''The following graph allows us to analyze how the "hour posted" influences the number of plays on each post. This analysis is conducted separately for each day of the week to isolate the impact of the day of the week on the number of plays.'''
# Create a dropdown selection for choosing the plot
selected_plot = st.selectbox('Select a Day to see average plays per hour posted of that day', ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# Display the selected plot based on the user's choice
if selected_plot == 'Sunday':
    st.plotly_chart(sunplaysplot)
elif selected_plot == 'Monday':
    st.plotly_chart(monplaysplot)
elif selected_plot == 'Tuesday':
    st.plotly_chart(tuesplaysplot)
elif selected_plot == 'Wednesday':
    st.plotly_chart(wedsplaysplot)
elif selected_plot == 'Thursday':
    st.plotly_chart(thursplaysplot)
elif selected_plot == 'Friday':
    st.plotly_chart(friplaysplot)
elif selected_plot == 'Saturday':
    st.plotly_chart(satplaysplot)

# Subtitle (using Markdown)
st.markdown("<h4>K-Means Cluster</h4>", unsafe_allow_html=True)


'''In order to group our data effectively, we employed the Elbow Method to determine the ideal number of clusters. The graph visually demonstrates the relationship between the number of clusters (K) and the variance within each cluster. Upon analysis, we observed a significant 'elbow' point in the graph, which occurs between K = 2 and K = 3. This clear inflection suggests that our data is best segmented into 3 distinct groups. 
'''
st.plotly_chart(elbow, use_container_width=False)

'''The graph depicts results from a K-means clustering model applied to our Reels dataset, which categorizes data points based on our three primary Key Performance Indicators (KPIs): Plays, Likes, and Comments. This approach enables us to gain insights into distinct performance groups. Additionally, by factoring in control variables, we can pinpoint the key influencers behind the formation of these groups.
'''
st.plotly_chart(cluster, use_container_width=True)


#Now we will do the same we did above but for photos data


# count how many photos where posted on each day of the week
postsperday = PhotosData.groupby('day_of_week')['index'].count()
# convert pandas series to dataframes
postdailycount = postsperday.to_frame()
# Set index as new column (day_of_week)
postdailycount.reset_index(inplace=True)
# 'index' column to 'dailycount'
postdailycount.rename({'index': 'dailycount'}, axis=1, inplace=True)

# set 'dailycount' column as a variable
dailycount = postdailycount['dailycount']

# get the likes counts for each day separately, not using this yet
saturdayposts = postdailycount.loc[postdailycount['day_of_week'] == 'Sat']

# get daily average plays
avgdailylikes = PhotosData.groupby('day_of_week')['likes'].mean()

# convert pandas series to dataframes
dailylikesmean = avgdailylikes.to_frame()

# Set index as new column (day_of_week)
dailylikesmean.reset_index(inplace=True)

# add daily count as a new column
dailylikesmean['dailycount'] = dailycount

# Reorder index according to day of the week
dailyplaysmean = dailylikesmean.reindex([1, 5, 6, 4, 0, 2, 3])


# Extract data from each day of the week separately
sunday = PhotosData.loc[PhotosData['day_of_week'] == 'Sun']
monday = PhotosData.loc[PhotosData['day_of_week'] == 'Mon']
tuesday = PhotosData.loc[PhotosData['day_of_week'] == 'Tues']
wednesday = PhotosData.loc[PhotosData['day_of_week'] == 'Weds']
thursday = PhotosData.loc[PhotosData['day_of_week'] == 'Thurs']
friday = PhotosData.loc[PhotosData['day_of_week'] == 'Fri']
saturday = PhotosData.loc[PhotosData['day_of_week'] == 'Sat']

# calculate the average of plays per hour each day
meanlikessun = sunday.groupby('hour posted')['likes'].mean()
meanlikesmon = monday.groupby('hour posted')['likes'].mean()
meanlikestues = tuesday.groupby('hour posted')['likes'].mean()
meanlikesweds = wednesday.groupby('hour posted')['likes'].mean()
meanlikesthurs = thursday.groupby('hour posted')['likes'].mean()
meanlikesfri = friday.groupby('hour posted')['likes'].mean()
meanlikessat = saturday.groupby('hour posted')['likes'].mean()

# convert pandas series to dataframes
sunlikes = meanlikessun.to_frame()
monlikes = meanlikesmon.to_frame()
tueslikes = meanlikestues.to_frame()
wedslikes = meanlikesweds.to_frame()
thurslikes = meanlikesthurs.to_frame()
frilikes = meanlikesfri.to_frame()
satlikes = meanlikessat.to_frame()

# Set index as new column (hour posted), so that it can be used in the barplot
sunlikes.reset_index(inplace=True)
monlikes.reset_index(inplace=True)
tueslikes.reset_index(inplace=True)
wedslikes.reset_index(inplace=True)
thurslikes.reset_index(inplace=True)
frilikes.reset_index(inplace=True)
satlikes.reset_index(inplace=True)

# calculate the count of likes per hour each day
postcountsun = sunday.groupby('hour posted')['likes'].count()
postcountmon = monday.groupby('hour posted')['likes'].count()
postcounttues = tuesday.groupby('hour posted')['likes'].count()
postcountweds = wednesday.groupby('hour posted')['likes'].count()
postcountthurs = thursday.groupby('hour posted')['likes'].count()
postcountfri = friday.groupby('hour posted')['likes'].count()
postcountsat = saturday.groupby('hour posted')['likes'].count()

# convert pandas series to dataframes
postcsun = postcountsun.to_frame()
postcmon = postcountmon.to_frame()
postctues = postcounttues.to_frame()
postcweds = postcountweds.to_frame()
postcthurs = postcountthurs.to_frame()
postcfri = postcountfri.to_frame()
postcsat = postcountsat.to_frame()

# Rename the column as 'count'
# Set index as new column (hour posted) in order to be able to add it as a new column to the plays charts
# Set the count column as a variable
# Add count column to each day's plays chart
postcsun.rename({'likes': 'count'}, axis=1, inplace=True)
postcsun.reset_index(inplace=True)
count = postcsun['count']
sunlikes['count'] = count

postcmon.rename({'likes': 'count'}, axis=1, inplace=True)
postcmon.reset_index(inplace=True)
count = postcmon['count']
monlikes['count'] = count

postctues.rename({'likes': 'count'}, axis=1, inplace=True)
postctues.reset_index(inplace=True)
count = postctues['count']
tueslikes['count'] = count

postcweds.rename({'likes': 'count'}, axis=1, inplace=True)
postcweds.reset_index(inplace=True)
count = postcweds['count']
wedslikes['count'] = count

postcthurs.rename({'likes': 'count'}, axis=1, inplace=True)
postcthurs.reset_index(inplace=True)
count = postcthurs['count']
thurslikes['count'] = count

postcfri.rename({'likes': 'count'}, axis=1, inplace=True)
postcfri.reset_index(inplace=True)
count = postcfri['count']
frilikes['count'] = count

postcsat.rename({'likes': 'count'}, axis=1, inplace=True)
postcsat.reset_index(inplace=True)
count = postcsat['count']
satlikes['count'] = count


# Bar plots for all variables
likesplot = px.bar(PhotosData, x="index", y="likes", color="day_of_week", title="Likes", hover_data=['shortcap', 'hour posted', 'hashtags'])
# likesplot.show()
commentsplot = px.bar(PhotosData, x="index", y="comments", color="day_of_week", title="Comments", hover_data=['shortcap', 'hour posted', 'hashtags'])
# commentsplot.show()

# Make a bar plots of average plays per day
dailylikesplot = px.bar(dailylikesmean, x="day_of_week", y="likes", title='Average Daily Likes', hover_data=['dailycount'])
# dailylikesplot.show()

# Make bar plots of average plays per hour each day
sunlikesplot = px.bar(sunlikes, x="hour posted", y="likes", title="Sunday", hover_data=['count'])
# sunlikesplot.show()
monlikesplot = px.bar(monlikes, x="hour posted", y="likes", title="Monday", hover_data=['count'])
# monlikesplot.show()
tueslikesplot = px.bar(tueslikes, x="hour posted", y="likes", title="Tuesday", hover_data=['count'])
# tueslikesplot.show()
wedslikesplot = px.bar(wedslikes, x="hour posted", y="likes", title="Wednesday", hover_data=['count'])
# wedslikesplot.show()
thurslikesplot = px.bar(thurslikes, x="hour posted", y="likes", title="Thursday", hover_data=['count'])
# thurslikesplot.show()
frilikesplot = px.bar(frilikes, x="hour posted", y="likes", title="Friday", hover_data=['count'])
# frilikesplot.show()
satlikesplot = px.bar(satlikes, x="hour posted", y="likes", title='Saturday', hover_data=['count'])
# satlikesplot.show()


# Subtitle (using Markdown)
st.markdown("<h2>Photos Data Analysis</h2>", unsafe_allow_html=True)


# Show all charts in streamlit
'''The following charts provide a comprehensive view of our photos' performance over time, with a focus on two key performance indicators (KPIs): Comments and Likes. These charts are meticulously color-coded based on the 'day of the week,' offering a visual tool to assess how different days impact each KPI. Through this thoughtful organization, we gain valuable insights into the trends and patterns that drive engagement and interaction with our content.
'''
st.plotly_chart(commentsplot, use_container_width=True)
st.plotly_chart(likesplot, use_container_width=True)

'''The following chart offers deeper insights into how the 'day of the week' influences our average likes. It allows us to examine how different days impact the average number of likes our photos receive. This analysis helps us identify trends and patterns related to content engagement throughout the week'''
st.plotly_chart(dailylikesplot, use_container_width=True)

'''The following graph allows us to analyze how the "hour posted" influences the number of likes on each post. This analysis is conducted separately for each day of the week to isolate the impact of the day of the week on the number of likes.'''
selected_plot_photos = st.selectbox('Select a Day to see average likes per hour posted of that day', ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# Display the selected plot based on the user's choice
if selected_plot_photos == 'Sunday':
    st.plotly_chart(sunlikesplot)
elif selected_plot_photos == 'Monday':
    st.plotly_chart(monlikesplot)
elif selected_plot_photos == 'Tuesday':
    st.plotly_chart(tueslikesplot)
elif selected_plot_photos == 'Wednesday':
    st.plotly_chart(wedslikesplot)
elif selected_plot_photos == 'Thursday':
    st.plotly_chart(thurslikesplot)
elif selected_plot_photos == 'Friday':
    st.plotly_chart(frilikesplot)
elif selected_plot_photos == 'Saturday':
    st.plotly_chart(satlikesplot)