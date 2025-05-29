from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned")
train_data = dataset["train"]

print(train_data[43235])
# print("\n" * 3)

# # Show the hardest samples
# print("THE VERY HARDEST SAMPLE:")
# for idx in [51510, 49802, 3887, 36743, 20179]:
#     row = train_data[idx]
#     print(f"Row {idx}:")
#     print(f"  instruction length: {len(row['instruction'])}")
#     print(f"  input length: {len(row['input'])}")
#     print(f"  output length: {len(row['output'])}")
#     print(row)
#     print()

# print("\n" * 3)

# # Show the easiest samples
# print("THE VERY EASIEST SAMPLE:")
# for idx in [46262, 22118, 22867, 49849, 18858]:
#     row = train_data[idx]
#     print(f"Row {idx}:")
#     print(f"  instruction length: {len(row['instruction'])}")
#     print(f"  input length: {len(row['input'])}")
#     print(f"  output length: {len(row['output'])}")
#     print(row)
#     print()

# print("\n" * 3)

# # Compute length statistics across the dataset
# instruction_lengths = []
# input_lengths = []
# output_lengths = []

# for row in train_data:
#     instruction_lengths.append(len(row["instruction"]))
#     input_lengths.append(len(row["input"]))
#     output_lengths.append(len(row["output"]))

# def print_stats(name, lengths):
#     print(f"{name} length stats:")
#     print(f"  Min:  {np.min(lengths)}")
#     print(f"  Max:  {np.max(lengths)}")
#     print(f"  Mean: {np.mean(lengths):.2f}")
#     print(f"  Median: {np.median(lengths):.2f}")
#     print()

# print("DATASET LENGTH STATISTICS:")
# print_stats("Instruction", instruction_lengths)
# print_stats("Input", input_lengths)
# print_stats("Output", output_lengths)

# # Count how many rows have a non-empty input
# non_empty_input_count = sum(1 for row in train_data if len(row["input"].strip()) > 0)
# total_rows = len(train_data)
# empty_input_count = total_rows - non_empty_input_count

# print(f"Total rows: {total_rows}")
# print(f"Rows with non-empty input: {non_empty_input_count}")
# print(f"Rows with empty input: {empty_input_count}")
# print(f"Percentage with non-empty input: {non_empty_input_count / total_rows * 100:.2f}%")

