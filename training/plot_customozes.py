import matplotlib.pyplot as plt
# Sample data
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
y2 = [1, 4, 6, 8, 10]

# Create a plot with customizations
plt.plot(x, y1, marker='o', label='Series 1', color='blue')
plt.plot(x, y2, marker='x', label='Series 2', color='green')

# Customizing the plot
plt.title("Customized Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()