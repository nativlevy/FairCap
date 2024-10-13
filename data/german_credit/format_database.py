import pandas as pd


def process_and_bin_data(input_file: str, output_file: str, bins: int = 10):
    # Load the data
    data = pd.read_csv(input_file)
    print("Data loaded successfully.")

    # Skip the first anonymous column by excluding it from binning operations
    data_no_anon = data.iloc[:, 1:]

    # Check for columns with more than 8 unique values
    columns_to_bin = [col for col in data_no_anon.columns if data_no_anon[col].nunique() > 8]
    print(f"Columns identified for binning: {columns_to_bin}")

    # Apply equal width binning to relevant columns
    for col in columns_to_bin:
        print(f"Processing column: {col}")

        # Check data type and skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(data_no_anon[col]):
            print(f"Skipping column '{col}' as it is non-numeric: {data_no_anon[col].dtype}")
            continue

        if col == "age":
            # Special handling for 'age' column: Convert bins to range strings
            age_bins = pd.cut(data_no_anon[col], bins=bins)
            data[col + '_binned'] = age_bins.apply(lambda x: f"{int(x.left)}-{int(x.right)}")
            print(f"Binned 'age' column: {data[col + '_binned'].unique()}")
        else:
            # General case for other numeric columns
            data[col + '_binned'] = pd.cut(data_no_anon[col], bins=bins)
            print(f"Binned '{col}' column: {data[col + '_binned'].unique()}")

    # Save the modified DataFrame to the output file
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


# Apply the function to the uploaded file and save the output
input_path = 'german_credit_data_new.csv'
output_path = 'german_credit_data_binned_output.csv'
process_and_bin_data(input_path, output_path)
