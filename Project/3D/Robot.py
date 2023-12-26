import time
import numpy as np
from scipy.stats import norm

# Simulation parameters
dt = 0.1  # Time step
num_steps = 100  # Number of steps

# Robot parameters
initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial pose [x, y, z, roll, pitch, yaw]
motion_noise = 0.2  # Motion noise
measurement_noise = 0.01  # Measurement noise

def move_robot(pose, u):
    x, y, z, roll, pitch, yaw = pose
    v, omega_x, omega_y, omega_z = u[:4]  # Take the first four values from the control input
    
    # Update roll, pitch, and yaw
    delta_roll = omega_x * dt
    delta_pitch = omega_y * dt
    delta_yaw = omega_z * dt
    
    # Update position
    new_pose = np.array([
        x + v * np.cos(yaw) * dt,
        y + v * np.sin(yaw) * dt,
        z,
        roll + delta_roll,
        pitch + delta_pitch,
        yaw + delta_yaw
    ])
    
    # Add motion noise
    new_pose += np.random.normal(0, motion_noise, 6)
    
    return new_pose



def generate_sensor_reading_3d(true_pose):
    x, y, z, _, _, yaw = true_pose
    # Simulate measurements with noise
    range_measurement = np.sqrt(x**2 + y**2 + z**2) + np.random.normal(0, measurement_noise)
    bearing_measurement = np.arctan2(y, x) + np.random.normal(0, measurement_noise)
    pitch_measurement = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)) + np.random.normal(0, measurement_noise)
    yaw_measurement = yaw + np.random.normal(0, measurement_noise)
    return np.array([range_measurement, bearing_measurement, pitch_measurement, 0.0, 0.0, yaw_measurement])

# Simulate robot motion and generate sensor readings
true_poses = []
sensor_readings = []

current_pose = initial_pose
for step in range(num_steps):
    # Simulate control input (linear velocity, and angular velocities around x, y, and z axes)
    control_input = np.array([0.1, 0.05, 0.02, 0.01])
    
    # Simulate robot motion
    current_pose = move_robot(current_pose, control_input)
    
    # Generate sensor reading
    sensor_reading = generate_sensor_reading_3d(current_pose)
    
    # Save true pose and sensor reading
    true_poses.append(current_pose)
    sensor_readings.append(sensor_reading)

    # Print the current pose and sensor reading with timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"Timestamp: {timestamp}, Step: {step + 1}, True Pose: {current_pose}, Sensor Reading: {sensor_reading}")
