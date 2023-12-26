import numpy as np
from Robot import generate_sensor_reading_3d, move_robot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

class NNKFLocalizer:
    def __init__(self, input_dim=6):  # Update input_dim to 6 for a 3D robot
        # Define the neural network model
        self.model = self.build_model(input_dim)
        self.scaler = StandardScaler()

    def build_model(self, input_dim):
        model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dropout(0.2),  # Adding dropout for regularization
            Dense(32, activation='relu'),
            Dense(6, activation='linear')  # Output dimension is 6 for a 3D robot
        ])
        optimizer = Adam(learning_rate=0.1)  # Adjust the learning rate if needed
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def train_model(self, input_data, target_data, epochs=10):
        input_data_scaled = self.scaler.fit_transform(input_data)
        self.model.fit(input_data_scaled, target_data, epochs=epochs, verbose=0)

    def localize(self, sensor_reading):
        scaled_sensor_reading = self.scaler.transform(np.array([sensor_reading]))
        # Predict the pose using the trained neural network
        estimated_pose = self.model.predict(scaled_sensor_reading)[0]
        return estimated_pose

if __name__ == "__main__":
    import time

    # Simulation parameters
    dt = 0.1  # Time step
    num_steps = 100  # Number of steps

    # Robot parameters
    initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial pose [x, y, z, roll, pitch, yaw]
    motion_noise = 0.2  # Motion noise
    measurement_noise = 0.01  # Measurement noise

    # Create an instance of the NNKF localizer
    nnkf_localizer = NNKFLocalizer(input_dim=6)  # Update input_dim to 6 for a 3D robot

    # Simulate robot motion and generate sensor readings
    true_poses = []
    sensor_readings = []
    estimated_poses = []

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

        # NNKF localization
        nnkf_localizer.train_model(np.array(sensor_readings), np.array(true_poses), epochs=10)
        estimated_pose = nnkf_localizer.localize(sensor_reading)
        estimated_poses.append(estimated_pose)

        # Print the current pose, sensor reading, estimated pose, and true pose with timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"Timestamp: {timestamp}, Step: {step + 1}, True Pose: {current_pose}, "
              f"Sensor Reading: {sensor_reading}, Estimated Pose: {estimated_pose}")

    # Print the final estimated and true poses
    print("\nFinal Estimated Pose:", estimated_poses[-1])
    print("Final True Pose:", true_poses[-1])
