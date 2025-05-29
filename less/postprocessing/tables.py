import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure non-LaTeX-compatible PGF output (since you don't need LaTeX formatting)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "pgf.rcfonts": False,
})

def plot_histogram(df, column, bucket_width=0.05, output_base="histogram", output_dir="."):
    """
    Plots a histogram of the given column in the DataFrame and saves it as PDF.
    
    Parameters:
    - df: pandas DataFrame
    - column: str, name of the column to plot
    - bucket_width: float, width of each bin
    - output_base: str, base filename for output files
    - output_dir: str, directory to save the plot (defaults to current directory)
    """
    min_val = df[column].min()
    max_val = df[column].max()
    bins = np.arange(min_val, max_val + bucket_width, bucket_width)

    plt.figure(figsize=(6, 4))
    plt.hist(df[column], bins=bins, color="steelblue")

    plt.xlabel(column.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column.capitalize()}")
    plt.grid(True)

    # Save to PDF in the specified directory
    plt.savefig(os.path.join(output_dir, f"{output_base}.pdf"))
    plt.close()

    print(f"Saved histogram to {os.path.join(output_dir, f'{output_base}.pdf')}")

def plot_score_with_percentiles(df, column="score", output_base="score_plot_percentiles", output_dir="."):
    """
    Plots a line graph of the 'score' column, using percentiles for the x-axis.
    
    Parameters:
    - df: pandas DataFrame
    - column: str, name of the column to plot
    - output_base: str, base filename for the output plot
    - output_dir: str, directory to save the plot (defaults to current directory)
    """
    # Sort the DataFrame by the 'score' column in descending order
    df_sorted = df.sort_values(by=column, ascending=False).reset_index(drop=True)

    # Calculate percentiles (this generates values from 100% down to 0%)
    percentiles = np.linspace(0, 100, len(df_sorted))

    # Plot the line graph
    plt.figure(figsize=(6, 4))
    plt.plot(percentiles, df_sorted[column], color="dodgerblue", label=column)
    plt.xlabel("Top percentage")
    plt.ylabel(column.capitalize())
    plt.title(f"Score vs Percentiles")
    plt.legend(loc="best")
    plt.grid(True)

    # Customize x-ticks to show every 10% from 0% to 100%
    plt.xticks(np.arange(0, 101, 10))

    # Save the plot as PDF in the specified directory
    plt.savefig(os.path.join(output_dir, f"{output_base}.pdf"))
    plt.close()

    print(f"Saved plot to {os.path.join(output_dir, f'{output_base}.pdf')}")

# --- Main ---
if __name__ == "__main__":
    # Load DataFrame
    csv_file_path = "less/selected_data/llama3_2_1B_instruct_warmup_steps_fixed/truthful_qa/sorted.csv"
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()  # Ensure no leading/trailing spaces in column names

    # Get the directory where the CSV file is located
    output_dir = os.path.dirname(csv_file_path)

    # Plot histogram and save it as a PDF
    plot_histogram(df, column="score", bucket_width=0.001, output_base="score_histogram", output_dir=output_dir)

    # Plot line graph and save it as a PDF
    plot_score_with_percentiles(df, column="score", output_base="score_line_plot", output_dir=output_dir)
