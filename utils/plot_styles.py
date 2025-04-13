"""
Standardized plot styles for prediction vs ground truth plots.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# Define consistent colors with better contrast
PREDICTION_COLOR = '#1f77b4'  # Blue
GROUND_TRUTH_COLOR = '#ff7f0e'  # Orange

# Define line styles
PREDICTION_LINESTYLE = '--'  # Dashed line for predictions
GROUND_TRUTH_LINESTYLE = '-'  # Solid line for ground truth

# Define labels
PREDICTION_LABEL = 'Prediction'
GROUND_TRUTH_LABEL = 'Ground Truth'

# Define a list of distinct colors for multiple channels
CHANNEL_COLORS = [
    '#e41a1c',  # Red
    '#377eb8',  # Blue
    '#4daf4a',  # Green
    '#984ea3',  # Purple
    '#ff7f00',  # Orange
    '#ffff33',  # Yellow
    '#a65628',  # Brown
    '#f781bf',  # Pink
    '#999999',  # Gray
    '#8dd3c7',  # Mint
    '#fdb462',  # Light orange
    '#b3de69'   # Light green
]

# Ensure we have enough colors for many channels
# If more channels than colors, colors will repeat
def get_channel_color(index):
    """
    Get a color for a specific channel index.
    Will cycle through the defined colors if index exceeds available colors.
    """
    return CHANNEL_COLORS[index % len(CHANNEL_COLORS)]

# Define sensor-specific styles
SENSOR_STYLES = {
    'emg': {
        'color': '#e41a1c',  # Bright red
        'marker': 'o',
        'linewidth': 1.5,
        'alpha': 0.9
    },
    'acc': {
        'color': '#377eb8',  # Blue
        'marker': 's',
        'linewidth': 1.5,
        'alpha': 0.9
    },
    'gyro': {
        'color': '#4daf4a',  # Green
        'marker': '^',
        'linewidth': 1.5,
        'alpha': 0.9
    }
}

# Define a function to get style based on sensor type
def get_sensor_style(sensor_name):
    """
    Get the appropriate style for a sensor based on its name.
    
    Args:
        sensor_name (str): The name of the sensor
        
    Returns:
        dict: Style dictionary for the sensor
    """
    sensor_type = None
    if 'emg' in sensor_name.lower():
        sensor_type = 'emg'
    elif 'acc' in sensor_name.lower():
        sensor_type = 'acc'
    elif 'gyro' in sensor_name.lower():
        sensor_type = 'gyro'
    else:
        # Default style if sensor type can't be determined
        return {'color': '#1f77b4', 'marker': 'o', 'linewidth': 1.5, 'alpha': 0.9}
    
    return SENSOR_STYLES[sensor_type]

# Set global matplotlib parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6)
})
