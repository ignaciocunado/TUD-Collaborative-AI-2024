import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def plot_1_run_results(csv_file : str, human_name : str):
    # Change hard coded values per run
    # csv_file = "../trust_logs/wilco/wilco_3_no_baseline.csv"
    # human_name = "wilco"

    # Read the CSV using semicolon as delimiter
    df = pd.read_csv(csv_file, delimiter=';')

    # Get the final tick from the "tick_nr" column
    final_tick = df['tick_nr'].max()

    # Sort the DataFrame by tick_nr
    df_sorted = df.sort_values(by='tick_nr')

    print("Total number of ticks to end game: ", str(final_tick))

    # Create flags for non-empty actions for each tick.
    # Empty actions are treated as 0 and non-empty as 1.
    df_sorted['rescuebot_action_flag'] = df_sorted['rescuebot_action'].apply(
        lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0)
    df_sorted[f'{human_name}_action_flag'] = df_sorted[f'{human_name}_action'].apply(
        lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0)

    # Group by tick_nr and sum the action flags for both actors.
    actions_per_tick = df_sorted.groupby('tick_nr', as_index=False).agg({
        'rescuebot_action_flag': 'sum',
        f'{human_name}_action_flag': 'sum'
    })

    # Compute cumulative sum for each actor so that the plot shows the total actions up to each tick.
    actions_per_tick['rescuebot_cumsum'] = actions_per_tick['rescuebot_action_flag'].cumsum()
    actions_per_tick[f'{human_name}_cumsum'] = actions_per_tick[f'{human_name}_action_flag'].cumsum()


    print("Starting to compute number of collaborative actions")
    # Plot number of collaborative actions
    df_sorted['collaborative_action_flag'] = df_sorted[f'{human_name}_action'].apply(
        lambda x: 1 if isinstance(x, str) and "Together" in x else 0
    )

    # Group by tick and sum the collaborative_action_flag
    collab_per_tick = df_sorted.groupby('tick_nr', as_index=False)['collaborative_action_flag'].sum()

    # Compute the cumulative sum of collaborative actions
    collab_per_tick['collab_cumsum'] = collab_per_tick['collaborative_action_flag'].cumsum()

    # Print the final total number of collaborative actions
    final_collab_actions = collab_per_tick['collab_cumsum'].iloc[-1] if not collab_per_tick.empty else 0
    print("Total number of collaborative actions:", final_collab_actions)

    # Plot the cumulative collaborative actions
    plt.figure(figsize=(10, 6))
    plt.plot(collab_per_tick['tick_nr'], collab_per_tick['collab_cumsum'],
             color='green', linestyle='-', linewidth=2, label='Collaborative Actions')
    plt.xlabel('Tick Number')
    plt.ylabel('Cumulative Collaborative Actions')
    plt.title('Cumulative Collaborative Actions (Containing "Together") per Tick')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Plot the cumulative actions for each actor:
    plt.figure(figsize=(10, 6))
    plt.plot(actions_per_tick['tick_nr'], actions_per_tick['rescuebot_cumsum'],
             color='red', linestyle='-', linewidth=2, label='Rescuebot')
    plt.plot(actions_per_tick['tick_nr'], actions_per_tick[f'{human_name}_cumsum'],
             color='blue', linestyle='-', linewidth=2, markersize=1, label=human_name.capitalize())
    plt.xlabel('Tick Number')
    plt.ylabel('Cumulative Number of Actions')
    plt.title('Cumulative Actions per Tick')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    rescuebot_ts = actions_per_tick[['tick_nr', 'rescuebot_cumsum']].copy()
    # 2) Human cumulative actions
    human_ts = actions_per_tick[['tick_nr', f'{human_name}_cumsum']].copy()
    # 3) Collaborative actions
    collab_ts = collab_per_tick[['tick_nr', 'collab_cumsum']].copy()

    return rescuebot_ts, human_ts, collab_ts

def plot_combined_results():
    """
    Plots and compares all 5 runs
    """
    # Five CSVs (placeholder paths)
    runs = [
        ("../trust_logs/wilco/wilco_1_no_baseline.csv", "No Baseline First Round"),
        ("../trust_logs/wilco/wilco_2_no_baseline.csv", "No Baseline Second Round"),
        ("../trust_logs/wilco/wilco_3_no_baseline.csv", "No Baseline Third Round"),
        ("../trust_logs/wilco/wilco_baseline_always_trust.csv", "Always Trust Baseline"),
        ("../trust_logs/wilco/wilco_baseline_never_trust.csv", "Never Trust Baseline"),
    ]

    # Lists to store time series data from each run
    rescuebot_series_list = []
    human_series_list = []
    collab_values = []  # We'll store the final collaborative action values here
    labels = []

    # Call get_time_series for each CSV
    for csv_file, label in runs:
        rescuebot_ts, human_ts, collab_ts = plot_1_run_results(csv_file, human_name="wilco")

        rescuebot_series_list.append(rescuebot_ts)
        human_series_list.append(human_ts)

        # Get the final collaborative value from the last row of 'collab_cumsum'
        final_collab_value = collab_ts['collab_cumsum'].iloc[-1] if not collab_ts.empty else 0
        collab_values.append(final_collab_value)

        labels.append(label)

    # ------------------------------------------------------
    # 1) Plot HUMAN actions for all runs in a single figure
    # ------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for i, human_ts in enumerate(human_series_list):
        # The second column should be the cumsum for the human
        human_cumsum_col = human_ts.columns[-1]
        plt.plot(human_ts['tick_nr'], human_ts[human_cumsum_col], label=labels[i])

    plt.xlabel('Tick Number')
    plt.ylabel('Cumulative Human Actions')
    plt.title('Human Actions Over Time (All Runs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # 2) Plot AGENT (Rescuebot) actions for all runs in one figure
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for i, rescuebot_ts in enumerate(rescuebot_series_list):
        rescuebot_cumsum_col = rescuebot_ts.columns[-1]
        plt.plot(rescuebot_ts['tick_nr'], rescuebot_ts[rescuebot_cumsum_col], label = labels[i])

    plt.xlabel('Tick Number')
    plt.ylabel('Cumulative Agent Actions')
    plt.title('Agent Actions Over Time (All Runs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------
    # 3) Compare final collaborative actions for the five runs in a bar chart
    # ----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.bar(labels, collab_values, color='green')
    plt.xlabel('Mission')
    plt.ylabel('Final Cumulative Collaborative Actions')
    plt.title('Comparison of Collaborative Actions Across Missions and Baselines')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_combined_results()