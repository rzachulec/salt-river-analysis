import matplotlib.pyplot as plt

# Sample data
times = ['08:00', '09:00', '10:00', '11:00', '12:00']  # Example time data
temp_celsius = [10, 12, 15, 18, 20]  # Example temperature data in Celsius

fig, ax1 = plt.subplots()

# Plotting temperature in Celsius
ax1.plot(times, temp_celsius, color='b', label='Temperature')
ax1.set_xlabel('Time')
ax1.grid(axis='y')
ax1.set_ylabel('Temperature (°C)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Creating a secondary y-axis for Fahrenheit with the same data points
def celsius_to_fahrenheit(x):
    return x * 9/5 + 32

def fahrenheit_to_celsius(x):
    return (x - 32) * 5/9

ax2 = ax1.twinx()
ax2.set_ylabel('Temperature (°F)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Set the limits for the secondary y-axis to match the Celsius scale
ax2.set_ylim(celsius_to_fahrenheit(ax1.get_ylim()[0]), celsius_to_fahrenheit(ax1.get_ylim()[1]))

# Ensure the tick marks correspond correctly between the two scales
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

plt.title('Temperature Over Time')
plt.show()