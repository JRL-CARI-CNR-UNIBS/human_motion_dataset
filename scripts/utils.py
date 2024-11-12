import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter


# Filter and differentiate the configuration data to get the velocities, accelerations, and jerks
def filt_and_diff_config(X, filter_window=11, sav_gol_order=3, delta_t=0.01, mode='nearest'):
    X_filt = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order, deriv=0, axis=0, mode=mode)  # filtered joint positions
    Xdot = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order, deriv=1, delta=delta_t, axis=0, mode=mode)    # joint velocities
    Xddot = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order, deriv=2, delta=delta_t, axis=0, mode=mode)   # joint accelerations
    Xdddot = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order, deriv=3, delta=delta_t, axis=0, mode=mode)  # joint jerks

    return X_filt, Xdot, Xddot, Xdddot


# FILTER AND DIFFERENTIATES KEYPOINTS AS WELL
# Filter and differentiate the configuration data to get the velocities, accelerations, and jerks
def filt_and_diff_config_kpts(X, kpt, filter_window=11, sav_gol_order=3, delta_t=0.01, mode='nearest'):
    X_filt = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order,
                           deriv=0, axis=0, mode=mode)                     # filtered joint positions
    Xdot = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order,
                         deriv=1, delta=delta_t, axis=0, mode=mode)        # joint velocities
    Xddot = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order,
                          deriv=2, delta=delta_t, axis=0, mode=mode)       # joint accelerations
    Xdddot = savgol_filter(X, window_length=filter_window, polyorder=sav_gol_order,
                           deriv=3, delta=delta_t, axis=0, mode=mode)      # joint jerks
    kpt_filt = savgol_filter(kpt, window_length=filter_window, polyorder=sav_gol_order,
                             deriv=0, axis=0, mode=mode)                   # filtered keypoints
    kptdot = savgol_filter(kpt, window_length=filter_window, polyorder=sav_gol_order,
                           deriv=1, delta=delta_t, axis=0, mode=mode)      # keypoints velocities
    kptddot = savgol_filter(kpt, window_length=filter_window, polyorder=sav_gol_order,
                            deriv=2, delta=delta_t, axis=0, mode=mode)     # keypoints accelerations

    return X_filt, Xdot, Xddot, Xdddot, kpt_filt, kptdot, kptddot


# Define helper function to build the configuration dataset, instruction_id, and velocity
def build_configuration_dataset(input_df, sub, instr, task, vel, 
                                conf_names, conf_names_filt, conf_names_vel, conf_names_acc, conf_names_jerk,
                                param_names, kpt_names,
                                filter_window=11, sav_gol_order=3, new_dt=0.01, var_to_plot=None, differentiate_cartesian=False,
                                kpt_names_filt=None, kpt_names_vel=None, kpt_names_acc=None):
    
    # Check arguments
    if var_to_plot:
        assert(isinstance(var_to_plot, str)), "'var_to_plot' must be a string"
    if differentiate_cartesian:
        assert(kpt_names_filt is not None and kpt_names_vel is not None and kpt_names_acc is not None), \
            "kpt_names_filt, kpt_names_vel, and kpt_names_acc must be provided if differentiate_cartesian is True"
    
    # Select the data for the current subject, instruction_id, task, and velocity
    idx = (input_df['Subject'] == sub) & (input_df['Instruction_id'] == instr) & (input_df['Task_name'] == task) & (input_df['Velocity'] == vel)
    df = input_df[idx]

    # Get the time array (relative time to the start of the trajectory in seconds)
    try:
        t = pd.to_datetime(df['Timestamp']) - pd.to_datetime(df['Timestamp'].iloc[0])
        t_sec = t.dt.total_seconds().values # convert from pandas timestamp to seconds to numpy array
    except:
        print(f"Error: subject {sub}, instruction {instr}, task {task}, and velocity {vel}")
        print(df)
        import sys
        sys.exit(0)

    # Get the configuration data (joint positions)
    X = df[conf_names].values

    # Resample the time array to a fixed time step
    t_sec_resampled = np.arange(t_sec[0], t_sec[-1], new_dt)

    # Interpolate the configuration, param, and keypoint data to the new time array
    X_interp = np.array([np.interp(t_sec_resampled, t_sec, X[:, i]) for i in range(X.shape[1])]).T
    param_interp = np.array([np.interp(t_sec_resampled, t_sec, df[param_names].values[:, i]) for i in range(len(param_names))]).T
    kpt_interp = np.array([np.interp(t_sec_resampled, t_sec, df[kpt_names].values[:, i]) for i in range(len(kpt_names))]).T

    # Filter and differentiate
    if not differentiate_cartesian:
        # JOINT-SPACE: (position, velocities, accelerations, and jerks)
        X_filt, Xdot, Xddot, Xdddot = filt_and_diff_config(X_interp, filter_window=filter_window,
                                                        sav_gol_order=sav_gol_order, delta_t=new_dt,
                                                        mode='nearest')
    else:
        # JOINT-SPACE: (position, velocities, accelerations, and jerks) + CARTESIAN-SPACE: (position, velocities, accelerations)
        X_filt, Xdot, Xddot, Xdddot, \
            kpt_filt, kptdot, kptddot = filt_and_diff_config_kpts(X_interp, kpt_interp,
                                                                  filter_window=filter_window,
                                                                  sav_gol_order=sav_gol_order, delta_t=new_dt,
                                                                  mode='nearest')
    
    # Possibly plot the configuration and velocity data
    if var_to_plot:
        # Select variable to plot
        plot_idx = conf_names.index(var_to_plot)
        print(f'Plotting configuration and velocity for {conf_names[plot_idx]} for subject {sub}')

        # select which cartesian variable to plot (wrist x,y,z position)
        plot_idxs_cartesian = None
        if 'right' in var_to_plot or 'left' in var_to_plot:
            kpt_number = 4 if 'right' in var_to_plot else 7
            plot_idxs_cartesian = [kpt_names.index(s) for s in [f'human_kp{kpt_number}_x', f'human_kp{kpt_number}_y', f'human_kp{kpt_number}_z']]
        else:
            plot_idxs_cartesian = None

        # Plot the filtered configuration, the velocity, the acceleration, the jerk, and the cartesian position of the wrist
        plot_differentiation(X_filt, Xdot, Xddot, Xdddot, t_sec_resampled,
                             sub, conf_names,
                             plot_idx, plot_idxs_cartesian=plot_idxs_cartesian,
                             kpt_filt=kpt_filt, kpt_vel=kptdot, kpt_acc=kptddot)

    # Create the DataFrame with interpolated data
    output_df = pd.DataFrame({
        'Time': t_sec_resampled,
        'Subject': [sub] * len(t_sec_resampled),
        'Instruction_id': [instr] * len(t_sec_resampled),
        'Task_name': [task] * len(t_sec_resampled),
        'Velocity': [vel] * len(t_sec_resampled)
    })

    # Create DataFrames for the matrix data columns
    kpt_df = pd.DataFrame(kpt_interp, columns=kpt_names)
    if differentiate_cartesian:
        kpt_filt = pd.DataFrame(kpt_filt, columns=kpt_names_filt)
        kpt_dot = pd.DataFrame(kptdot, columns=kpt_names_vel)
        kpt_ddot = pd.DataFrame(kptddot, columns=kpt_names_acc)
    param_df = pd.DataFrame(param_interp, columns=param_names)
    conf_df = pd.DataFrame(X_interp, columns=conf_names)
    filt_df = pd.DataFrame(X_filt, columns=conf_names_filt)
    vel_df = pd.DataFrame(Xdot, columns=conf_names_vel)
    acc_df = pd.DataFrame(Xddot, columns=conf_names_acc)
    jerk_df = pd.DataFrame(Xdddot, columns=conf_names_jerk)

    # Concatenate all DataFrames along the columns
    if not differentiate_cartesian:
        output_df = pd.concat([output_df, kpt_df, param_df, conf_df, filt_df, vel_df, acc_df, jerk_df], axis=1)
    else:
        output_df = pd.concat([output_df, kpt_df, kpt_filt, kpt_dot, kpt_ddot, param_df,  # type: ignore
                               conf_df, filt_df, vel_df, acc_df, jerk_df], axis=1) # type: ignore

    return output_df


# Plot the X_filt and Xdot to check the differentiation using plotly
def plot_differentiation(X_filt, Xdot, Xddot, Xdddot, t_sec,
                         subject, conf_names, 
                         plot_idx, plot_idxs_cartesian=None, kpt_filt=None, kpt_vel=None, kpt_acc=None):

    if not plot_idxs_cartesian:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=t_sec, y=X_filt[:, plot_idx], mode='markers', name='q(t) (filtered)', showlegend=False),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=t_sec, y=Xdot[:, plot_idx], mode='markers', name='q_dot(t)', showlegend=False),
                    row=2, col=1)
        fig.add_trace(go.Scatter(x=t_sec, y=Xddot[:, plot_idx], mode='markers', name='q_ddot(t)', showlegend=False),
                    row=3, col=1)
        fig.add_trace(go.Scatter(x=t_sec, y=Xdddot[:, plot_idx], mode='markers', name='q_dddot(t)', showlegend=False),
                    row=4, col=1)
        
        fig.update_yaxes(title_text='q (filtered) [rad]', row=1, col=1)
        fig.update_yaxes(title_text='q_dot [rad]', row=2, col=1)
        fig.update_yaxes(title_text='q_ddot [rad]', row=3, col=1)
        fig.update_yaxes(title_text='q_dddot [rad]', row=4, col=1)
        
        # Update x-axes for all rows
        for i in range(1, 5):
            fig.update_xaxes(title_text='time [s]', range=[t_sec[0], t_sec[-1]], row=i, col=1, showticklabels=True)

        
    else:
        assert(kpt_filt is not None and kpt_vel is not None and kpt_acc is not None), \
            "kpt_filt, kpt_vel, and kpt_acc must be provided if plot_idxs_cartesian is not None"

        fig = make_subplots(rows=7, cols=1, shared_xaxes=True)

        var_name = ['pos', 'vel', 'acc']
        units = ['[m]', '[m/s]', '[m/s^2]']

        wrist_x = np.vstack((kpt_filt[:, plot_idxs_cartesian[0]],
                             kpt_vel[:, plot_idxs_cartesian[0]],
                             kpt_acc[:, plot_idxs_cartesian[0]])).T
        wrist_y = np.vstack((kpt_filt[:, plot_idxs_cartesian[1]],
                             kpt_vel[:, plot_idxs_cartesian[1]],
                             kpt_acc[:, plot_idxs_cartesian[1]])).T
        wrist_z = np.vstack((kpt_filt[:, plot_idxs_cartesian[2]],
                             kpt_vel[:, plot_idxs_cartesian[2]],
                             kpt_acc[:, plot_idxs_cartesian[2]])).T
        
        # Loop through each column of wrist_x and add a trace
        for i in range(wrist_x.shape[1]):
            fig.add_trace(go.Scatter(x=t_sec, y=wrist_x[:, i], mode='markers', name=f'wrist x {var_name[i]} {units[i]}', showlegend=True),
                          row=1, col=1)
            
        for i in range(wrist_y.shape[1]):
            fig.add_trace(go.Scatter(x=t_sec, y=wrist_y[:, i], mode='markers', name=f'wrist y {var_name[i]} {units[i]}', showlegend=True),
                          row=2, col=1)
            
        for i in range(wrist_z.shape[1]):
            fig.add_trace(go.Scatter(x=t_sec, y=wrist_z[:, i], mode='markers', name=f'wrist z {var_name[i]} {units[i]}', showlegend=True),
                          row=3, col=1)
            
        fig.add_trace(go.Scatter(x=t_sec, y=X_filt[:, plot_idx], mode='markers', name='q(t) (filtered)', showlegend=False),
                    row=4, col=1)
        fig.add_trace(go.Scatter(x=t_sec, y=Xdot[:, plot_idx], mode='markers', name='q_dot(t)', showlegend=False),
                    row=5, col=1)
        fig.add_trace(go.Scatter(x=t_sec, y=Xddot[:, plot_idx], mode='markers', name='q_ddot(t)', showlegend=False),
                    row=6, col=1)
        fig.add_trace(go.Scatter(x=t_sec, y=Xdddot[:, plot_idx], mode='markers', name='q_dddot(t)', showlegend=False),
                    row=7, col=1)

        fig.update_yaxes(title_text='wrist x',            row=1, col=1)
        fig.update_yaxes(title_text='wrist y',            row=2, col=1)
        fig.update_yaxes(title_text='wrist z',            row=3, col=1)        
        fig.update_yaxes(title_text='q (filtered) [rad]', row=4, col=1)
        fig.update_yaxes(title_text='q_dot [rad]',        row=5, col=1)
        fig.update_yaxes(title_text='q_ddot [rad]',       row=6, col=1)
        fig.update_yaxes(title_text='q_dddot [rad]',      row=7, col=1)
        
        # Update x-axes for all rows
        for i in range(1, 8):
            fig.update_xaxes(title_text='time [s]', range=[t_sec[0], t_sec[-1]], row=i, col=1, showticklabels=True)

    fig.update_layout(
        title=f'{conf_names[plot_idx]} vs time for subject {subject}',
        height=1500
    )

    fig.update_traces(marker=dict(size=3))
    fig.show()