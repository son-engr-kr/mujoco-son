#!/usr/bin/env python3
"""
Test script for compliant muscle implementation in MuJoCo.
Compares compliant muscle vs standard muscle behavior.
"""

import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def test_compliant_muscle():
    """Test compliant muscle implementation"""
    
    # Load the model
    model = mujoco.MjModel.from_xml_path("model_compliant_muscle_test.xml")
    data = mujoco.MjData(model)
    
    print("=== Compliant Muscle Test ===")
    print(f"Model loaded: {model.nq} DOFs, {model.nu} actuators")
    print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")
    print(f"Actuator gain types: {[model.actuator(i).gaintype for i in range(model.nu)]}")
    print(f"Actuator dyn types: {[model.actuator(i).dyntype for i in range(model.nu)]}")
    
    # Initialize compliant muscle states
    if hasattr(mujoco, 'mju_compliantMuscleInit'):
        mujoco.mju_compliantMuscleInit(model, data)
        print("Compliant muscle states initialized")
    
    # Simulation parameters
    n_steps = 1000
    dt = model.opt.timestep
    total_time = n_steps * dt
    
    # Storage arrays
    time_points = np.zeros(n_steps)
    pos_c = np.zeros(n_steps)  # compliant muscle mass position
    pos_r = np.zeros(n_steps)  # standard muscle mass position
    vel_c = np.zeros(n_steps)
    vel_r = np.zeros(n_steps)
    len_c = np.zeros(n_steps)  # tendon length
    len_r = np.zeros(n_steps)
    frc_c = np.zeros(n_steps)  # muscle force
    frc_r = np.zeros(n_steps)
    act_c = np.zeros(n_steps)  # activation
    act_r = np.zeros(n_steps)
    
    # Control signals (step input)
    ctrl_c = np.zeros(n_steps)
    ctrl_r = np.zeros(n_steps)
    
    # Apply step input at t=0.1s
    step_start = int(0.1 / dt)
    ctrl_c[step_start:] = 0.5  # 50% activation
    ctrl_r[step_start:] = 0.5   # 50% activation
    
    print(f"\nRunning simulation for {total_time:.3f}s...")
    print(f"Step input applied at t={step_start*dt:.3f}s")
    
    # Run simulation
    for i in range(n_steps):
        # Set control
        data.ctrl[0] = ctrl_c[i]  # compliant muscle
        data.ctrl[1] = ctrl_r[i]  # standard muscle
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Store data
        time_points[i] = i * dt
        pos_c[i] = data.qpos[0]  # slide_c joint position
        pos_r[i] = data.qpos[1]  # slide_r joint position
        vel_c[i] = data.qvel[0]  # slide_c joint velocity
        vel_r[i] = data.qvel[1]  # slide_r joint velocity
        len_c[i] = data.sensordata[0]  # tendon_c length
        len_r[i] = data.sensordata[1]  # tendon_r length
        frc_c[i] = data.sensordata[2]  # compliant_muscle_c force
        frc_r[i] = data.sensordata[3]  # standard_muscle_r force
        # act layout: the compliant_mtu actuator carries two activation variables,
        # [fiber_length, activation], so its activation is the LAST of its own slots.
        # The standard muscle that follows it has a single one.
        c_adr = model.actuator_actadr[0] + model.actuator_actnum[0] - 1
        r_adr = model.actuator_actadr[1] + model.actuator_actnum[1] - 1
        act_c[i] = data.act[c_adr] if c_adr >= 0 else 0
        act_r[i] = data.act[r_adr] if r_adr >= 0 else 0
        
        # Print progress
        if i % 100 == 0:
            print(f"t={i*dt:.3f}s: pos_c={pos_c[i]:.4f}, pos_r={pos_r[i]:.4f}, "
                  f"frc_c={frc_c[i]:.2f}, frc_r={frc_r[i]:.2f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Position comparison
    axes[0, 0].plot(time_points, pos_c, 'b-', label='Compliant Muscle', linewidth=2)
    axes[0, 0].plot(time_points, pos_r, 'r-', label='Standard Muscle', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('Mass Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Velocity comparison
    axes[0, 1].plot(time_points, vel_c, 'b-', label='Compliant Muscle', linewidth=2)
    axes[0, 1].plot(time_points, vel_r, 'r-', label='Standard Muscle', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Mass Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Force comparison
    axes[0, 2].plot(time_points, frc_c, 'b-', label='Compliant Muscle', linewidth=2)
    axes[0, 2].plot(time_points, frc_r, 'r-', label='Standard Muscle', linewidth=2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Force (N)')
    axes[0, 2].set_title('Muscle Force')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Tendon length comparison
    axes[1, 0].plot(time_points, len_c, 'b-', label='Compliant Muscle', linewidth=2)
    axes[1, 0].plot(time_points, len_r, 'r-', label='Standard Muscle', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Tendon Length (m)')
    axes[1, 0].set_title('Tendon Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Activation comparison
    axes[1, 1].plot(time_points, act_c, 'b-', label='Compliant Muscle', linewidth=2)
    axes[1, 1].plot(time_points, act_r, 'r-', label='Standard Muscle', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Activation')
    axes[1, 1].set_title('Muscle Activation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Control signal
    axes[1, 2].plot(time_points, ctrl_c, 'b-', label='Compliant Muscle', linewidth=2)
    axes[1, 2].plot(time_points, ctrl_r, 'r-', label='Standard Muscle', linewidth=2)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Control Signal')
    axes[1, 2].set_title('Control Input')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('compliant_muscle_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print final statistics
    print(f"\n=== Final Results ===")
    print(f"Compliant muscle - Final position: {pos_c[-1]:.4f}m, Final force: {frc_c[-1]:.2f}N")
    print(f"Standard muscle - Final position: {pos_r[-1]:.4f}m, Final force: {frc_r[-1]:.2f}N")
    print(f"Position difference: {abs(pos_c[-1] - pos_r[-1]):.4f}m")
    print(f"Force difference: {abs(frc_c[-1] - frc_r[-1]):.2f}N")
    
    return model, data, time_points, pos_c, pos_r, frc_c, frc_r

    

def visualize_simulation():
    """Visualize the simulation in real-time"""
    model = mujoco.MjModel.from_xml_path("model_compliant_muscle_test.xml")
    data = mujoco.MjData(model)
    
    # Initialize compliant muscle states
    if hasattr(mujoco, 'mju_compliantMuscleInit'):
        mujoco.mju_compliantMuscleInit(model, data)
    
    with mujoco_viewer.MujocoViewer(model, data) as viewer:
        step = 0
        while viewer.is_alive:
            # Apply step input after 100 steps
            if step > 100:
                data.ctrl[0] = 0.5  # compliant muscle
                data.ctrl[1] = 0.5  # standard muscle
            
            mujoco.mj_step(model, data)
            viewer.render()
            step += 1

if __name__ == "__main__":
    print("Testing Compliant Muscle Implementation")
    print("=" * 50)
    
    try:
        # Run the test
        model, data, time_points, pos_c, pos_r, frc_c, frc_r = test_compliant_muscle()
        
        print("\nTest completed successfully!")
        print("Results saved to 'compliant_muscle_test_results.png'")
        print("For low-level MTU internals, run: python compliant_analyze.py --show")
        
        # Ask if user wants to visualize
        response = input("\nDo you want to visualize the simulation? (y/n): ")
        if response.lower() == 'y':
            print("Starting visualization...")
            visualize_simulation()
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
