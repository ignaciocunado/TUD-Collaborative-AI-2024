import os

import pandas as pd
from matplotlib import pyplot as plt

def main():
    csv_file = pd.read_csv('trust_logs/trust_beliefs_per_tick.csv') # Enter the name here
    # Read the CSV file; note the delimiter is ';'
    try:
        data = pd.read_csv(csv_file, delimiter=";")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Create the plot with continuous lines for both data series
    plt.figure(figsize=(8, 5))
    plt.plot(data["Tick"], data["Willingness"], label="Willingness", linestyle='-')
    plt.plot(data["Tick"], data["Competence"], label="Competence", linestyle='-')

    # Add labels and title
    plt.xlabel("Tick")
    plt.ylabel("Value")
    plt.title("Competence and Willingness Over Ticks")
    plt.legend()
    plt.grid(True)

    # Determine the folder of the CSV file and define the output image path
    output_folder = os.path.dirname(os.path.abspath(csv_file))
    output_file = os.path.join(output_folder, "trust_beliefs_plot.png")

    # Save the plot image
    plt.savefig(output_file)
    print(f"Plot saved successfully as: {output_file}")

if __name__ == "__main__":
    main()