import numpy as np
from scipy.linalg import block_diag
from Robot import generate_sensor_reading, move_robot

class EKFLocalizer:
    def __init__(self, initial_pose, motion_noise, measurement_noise):
        self.mu = initial_pose  # Initial estimate of the robot pose [x, y, theta]
        self.sigma = np.zeros((3, 3))  # Initial covariance matrix
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise

    def predict(self, u, dt):
        # Predict the next state using the motion model (constant velocity)
        v, omega = u
        delta_theta = omega * dt

        # Jacobian of the motion model
        G = np.array([
            [1, 0, -v * np.sin(self.mu[2]) * dt],
            [0, 1, v * np.cos(self.mu[2]) * dt],
            [0, 0, 1]
        ])

        # Motion model
        motion_model = np.array([
            v * np.cos(self.mu[2]) * dt,
            v * np.sin(self.mu[2]) * dt,
            delta_theta
        ])

        # Update the state and covariance prediction
        self.mu = self.mu + motion_model
        self.sigma = G @ self.sigma @ G.T + np.diag([self.motion_noise, self.motion_noise, 0])

    def update(self, z):
        # Measurement model
        h = np.array([
            np.sqrt(self.mu[0]**2 + self.mu[1]**2),
            np.arctan2(self.mu[1], self.mu[0]),
            self.mu[2]
        ])

        # Jacobian of the measurement model
        H = np.array([
            [self.mu[0] / np.sqrt(self.mu[0]**2 + self.mu[1]**2), self.mu[1] / np.sqrt(self.mu[0]**2 + self.mu[1]**2), 0],
            [-self.mu[1] / (self.mu[0]**2 + self.mu[1]**2), self.mu[0] / (self.mu[0]**2 + self.mu[1]**2), 0],
            [0, 0, 1]
        ])

        # Measurement noise
        R = np.diag([self.measurement_noise, self.measurement_noise, self.measurement_noise])

        # Kalman gain
        K = self.sigma @ H.T @ np.linalg.inv(H @ self.sigma @ H.T + R)

        # Update the state and covariance estimate
        self.mu = self.mu + K @ (z - h)
        self.sigma = (np.eye(3) - K @ H) @ self.sigma

    # Modify the EKF localization method in ekf_localizer.py
    def localize(self, control_input, measurement, true_poses):
        dt = 0.1  # Time step (assuming the same time step as the motion model)

        # Prediction step
        self.predict(control_input, dt)

        # Update step
        self.update(measurement)

        # Return the estimated pose and true pose
        return self.mu, true_poses[-1]


if __name__ == "__main__":
    # Import necessary modules
    import time
    import numpy as np
    from scipy.stats import norm

    # Simulation parameters
    dt = 0.1  # Time step
    num_steps = 10000  # Number of steps

    # Robot parameters
    initial_pose = np.array([0.0, 0.0, 0.0])  # Initial pose [x, y, theta]
    motion_noise = 0.2  # Motion noise
    measurement_noise = 0.01  # Measurement noise

    # Create an instance of the EKF localizer
    ekf_localizer = EKFLocalizer(initial_pose, motion_noise, measurement_noise)

    # Simulate robot motion and generate sensor readings
    true_poses = []
    sensor_readings = []
    estimated_poses = []

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

        # EKF localization
        estimated_pose, true_pose = ekf_localizer.localize(control_input, sensor_reading)
        estimated_poses.append(estimated_pose)

        # Print the current pose, sensor reading, estimated pose, and true pose with timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"Timestamp: {timestamp}, Step: {step + 1}, True Pose: {current_pose}, "
              f"Sensor Reading: {sensor_reading}, Estimated Pose: {estimated_pose}")

    # Print the final estimated and true poses
    print("\nFinal Estimated Pose:", estimated_poses[-1])
    print("Final True Pose:", true_poses[-1])
