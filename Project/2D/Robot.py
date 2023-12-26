import time
import numpy as np
from scipy.stats import norm

# Simulation parameters
dt = 0.1  # Time step
num_steps = 100  # Number of steps

# Robot parameters
initial_pose = np.array([0.0, 0.0, 0.0])  # Initial pose [x, y, theta]
motion_noise = 0.2  # Motion noise
measurement_noise = 0.01  # Measurement noise

# Function to simulate robot motion
def move_robot(pose, u):
    x, y, theta = pose
    v, omega = u
    delta_theta = omega * dt
    new_pose = np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + delta_theta
    ])
    # Add motion noise
    new_pose += np.random.normal(0, motion_noise, 3)
    return new_pose

# Function to generate sensor readings
def generate_sensor_reading(true_pose):
    x, y, theta = true_pose
    # Simulate measurements with noise
    range_measurement = np.sqrt(x**2 + y**2) + np.random.normal(0, measurement_noise)
    bearing_measurement = np.arctan2(y, x) + np.random.normal(0, measurement_noise)
    theta_measurement = theta + np.random.normal(0, measurement_noise)
    return np.array([range_measurement, bearing_measurement, theta_measurement])

# Simulate robot motion and generate sensor readings
true_poses = []
sensor_readings = []

current_pose = initial_pose
for step in range(num_steps):
    # Simulate control input (linear velocity and angular velocity)
    control_input = np.array([0.1, 0.05])
    
    # Simulate robot motion
    current_pose = move_robot(current_pose, control_input)
    
    # Generate sensor reading
    sensor_reading = generate_sensor_reading(current_pose)
    
    # Save true pose and sensor reading
    true_poses.append(current_pose)
    sensor_readings.append(sensor_reading)

    # Print the current pose and sensor reading with timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"Timestamp: {timestamp}, Step: {step + 1}, True Pose: {current_pose}, Sensor Reading: {sensor_reading}")

