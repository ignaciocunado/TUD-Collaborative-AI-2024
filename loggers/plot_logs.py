import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def plot_1_run_results():
    # Change hard coded values per run
    csv_file = "logs/exp_strong_at_time_18h-24m-14s_date_05d-03m-2025y/world_1/actions__2025-03-05_182415.csv"
    human_name = "alice"

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


if __name__ == '__main__':
    plot_1_run_results()