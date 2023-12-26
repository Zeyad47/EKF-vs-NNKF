import time
import numpy as np
import matplotlib.pyplot as plt
from Robot import generate_sensor_reading_3d, move_robot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from scipy.linalg import block_diag
from ekf_localizer import EKFLocalizer 
from nnkf_localizer import NNKFLocalizer  
from sklearn.metrics import mean_squared_error

def run_simulation(num_steps):
    # Robot parameters
    initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial pose [x, y, z, roll, pitch, yaw]
    motion_noise = 0.2
    measurement_noise = 0.01

    # Create instances of the localizers
    ekf_localizer = EKFLocalizer(initial_pose, motion_noise, measurement_noise)
    nnkf_localizer = NNKFLocalizer(input_dim=6)  # Update input_dim to 6 for a 3D robot

    # Lists to store results
    true_poses = []
    ekf_estimated_poses = []
    nnkf_estimated_poses = []

    # Simulate robot motion and generate sensor readings
    current_pose = initial_pose
    for step in range(num_steps):
        # Simulate control input (linear velocity, and angular velocities around x, y, and z axes)
        control_input = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002])

        # Simulate robot motion
        current_pose = move_robot(current_pose, control_input)

        # Generate sensor reading
        sensor_reading = generate_sensor_reading_3d(current_pose)

        # Save true pose
        true_poses.append(current_pose)

        # EKF localization
        start_time = time.time()
        estimated_pose, _ = ekf_localizer.localize(control_input, sensor_reading, true_poses)
        ekf_estimated_poses.append(estimated_pose)
        ekf_time = time.time() - start_time

        # NNKF localization
        # Ensure the scaler is fitted before calling localize
        nnkf_localizer.train_model(np.array([sensor_reading]), np.array([current_pose]), epochs=10)
        start_time = time.time()
        estimated_pose_nn = nnkf_localizer.localize(sensor_reading)
        nnkf_estimated_poses.append(estimated_pose_nn)
        nnkf_time = time.time() - start_time

    # Calculate accuracy using Mean Squared Error (MSE)
    mse_ekf = mean_squared_error(true_poses, ekf_estimated_poses)
    mse_nnkf = mean_squared_error(true_poses, nnkf_estimated_poses)

    return true_poses, ekf_estimated_poses, nnkf_estimated_poses, ekf_time, nnkf_time, mse_ekf, mse_nnkf

if __name__ == "__main__":
    num_steps = int(input("Enter the number of steps for simulation: "))

    true_poses, ekf_estimated_poses, nnkf_estimated_poses, ekf_time, nnkf_time, mse_ekf, mse_nnkf = run_simulation(num_steps)

    # Plot the results
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.title('EKF Estimate vs True Pose (X-axis)')
    plt.plot(np.array(true_poses)[:, 0], label='True X')
    plt.plot(np.array(ekf_estimated_poses)[:, 0], label='EKF Estimated X')
    plt.legend()

    plt.subplot(132)
    plt.title('NNKF Estimate vs True Pose (X-axis)')
    plt.plot(np.array(true_poses)[:, 0], label='True X')
    plt.plot(np.array(nnkf_estimated_poses)[:, 0], label='NNKF Estimated X')
    plt.legend()

    plt.subplot(133)
    plt.title('EKF vs NNKF vs True Pose (X-axis)')
    plt.plot(np.array(true_poses)[:, 0], label='True X')
    plt.plot(np.array(ekf_estimated_poses)[:, 0], label='EKF Estimated X')
    plt.plot(np.array(nnkf_estimated_poses)[:, 0], label='NNKF Estimated X')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # Print the results
    print(f"\nEKF Time: {ekf_time} seconds")
    print(f"NNKF Time: {nnkf_time} seconds")

    print(f"\nEKF MSE: {mse_ekf}")
    print(f"NNKF MSE: {mse_nnkf}")
