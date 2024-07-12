import os
import pandas as pd
import configparser


# Load configuration from file
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


# Function to read the Excel file and get variable values
def read_variable_values_from_excel(filename):
    df = pd.read_excel(filename)
    variables = {col: df[col].dropna().astype(str).tolist() for col in df.columns}
    return variables


# Function to read results from the Excel file
def read_results_from_excel(filename):
    xls = pd.ExcelFile(filename)
    results = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        results.extend(df.to_dict(orient='records'))
    return results


# Function to get the school columns
def get_school_columns(df, variable_categories):
    school_columns = [col for col in df.columns if col not in variable_categories]
    return school_columns


# Function to create dataframe for results
def create_dataframe(results, variables, school_columns, top_n):
    # Initialize dataframe structure
    columns = school_columns[:top_n] + list(variables.keys())
    data = []

    for result in results:
        row = []
        for school in school_columns[:top_n]:
            row.append(result.get(school, ""))
        for variable in variables.keys():
            row.append(1 if result.get(variable) else 0)
        data.append(row)

    return pd.DataFrame(data, columns=columns)


# Main function
def main():
    config = load_config("config.ini")
    dependencies_folder = config.get("Directories", "dependencies")

    variables_path = os.path.join(dependencies_folder, config.get("Files", "variables"))
    results_path = os.path.join(dependencies_folder, config.get("Files", "results"))
    results_folder = config.get("Directories", "results_folder")
    top_n = config.getint("Processing", "top_n")

    # Read variables
    variables = read_variable_values_from_excel(variables_path)

    # Read results
    results = read_results_from_excel(results_path)

    # Get variable categories from the keys of the variables dictionary
    variable_categories = variables.keys()

    # Get school columns
    sample_df = pd.read_excel(results_path, sheet_name=0)
    school_columns = get_school_columns(sample_df, variable_categories)

    # Create dataframe for results
    df = create_dataframe(results, variables, school_columns, top_n)

    # Write dataframe to Excel file
    output_file = os.path.join(results_folder, "structured_results.xlsx")
    df.to_excel(output_file, index=False)

    print(f"Structured data written to {output_file}")


if __name__ == "__main__":
    main()
