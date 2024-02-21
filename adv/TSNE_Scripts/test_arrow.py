import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data
data = {
    "x": [1, 3, 5, 4],
    "y": [2, 4, 1, 5],
    "Alpha": [0.2, 0.5, 0.8, 0.4],
    "Sample Type": ["A", "B", "A", "B"]
}
df = pd.DataFrame(data)

# Create the scatterplot
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x="x", y="y", alpha=df['Alpha'], hue=df['Sample Type'], palette='Set1', ax=ax1)

# Define the points through which the line should pass
x_points = [1, 2, 5]
y_points = [2, 3, 1]

# Draw a line connecting all the points
ax1.plot(x_points[:-1], y_points[:-1], color='black')

# Add an arrow passing from the second-to-last to the last point
x_start = x_points[-2]
y_start = y_points[-2]
x_end = x_points[-1]
y_end = y_points[-1]
arrow_props = dict(arrowstyle='-|>', color='black')
ax1.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start), arrowprops=arrow_props)

# Save the plot as a PNG image file
plt.savefig('scatterplot_with_line_and_arrow.png')
