import os
import pandas as pd
import configparser
import scipy.stats as stats
from sklearn.decomposition import PCA

# Used to name variables appropriately
DEFAULTS = {
    'ethnicity': 'None',
    'income': 'family',
    'child ability': 'None',
    'child gender': 'kid',
    'child education level': 'K level',
    'parent education': 'None',
    'location': 'None'
}

# Load configuration from file
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


# Check which columns are schools
def get_school_columns(df):
    school_columns = []
    for col in df.columns[1:]:  # Skip the first column
        if df[col].apply(lambda x: isinstance(x, (int, float))).all():
            school_columns.append(col)
    return school_columns


# Check if a string is numeric
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Get variable name from a row based on the first non-default, non-blank, and non-numeric string
def get_variable_name(row, variable_columns):
    for col in variable_columns:
        value = row[col]
        if pd.notna(value) and str(value) != DEFAULTS.get(col, 'None') and str(value).strip() != '' and not is_numeric(
                value):
            return str(value).strip()
    return None


# Create symmetrical matrix for each school
def create_symmetrical_matrix(df, school, variable_names):
    matrix = pd.DataFrame(index=variable_names, columns=variable_names)
    for var1 in variable_names:
        for var2 in variable_names:
            idx1 = df.index[df['Variable'] == var1].tolist()[0]
            idx2 = df.index[df['Variable'] == var2].tolist()[0]
            val = abs(df.at[idx1, school] - df.at[idx2, school])
            matrix.at[var1, var2] = val

    # Add a special bottom row with the sum of each column
    sum_row = matrix.sum(axis=0)
    matrix.loc['Total'] = sum_row

    return matrix


# Calculate statistics and add to the matrix
def add_statistics_to_matrix(matrix, matrices):
    total_sums = [mtx.loc['Total'].sum() for mtx in matrices]
    num_columns = len(matrix.columns)
    mean_total = sum(total_sums) / (len(total_sums) * num_columns)
    std_total = pd.Series(matrix.loc['Total']).std()

    # Create a dataframe for statistical rows
    stats_df = pd.DataFrame(index=['', 'Mean of Totals', 'Std Dev of Totals', 'P-Value', 'Significance at 10%'],
                            columns=matrix.columns)
    stats_df.loc['Mean of Totals'] = [mean_total] * len(matrix.columns)
    stats_df.loc['Std Dev of Totals'] = [std_total] * len(matrix.columns)

    p_value_row = pd.Series(index=matrix.columns)
    significance_row = pd.Series(index=matrix.columns)
    for col in matrix.columns:
        z_stat = ((matrix.loc['Total', col] - mean_total) / std_total)
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        p_value_row[col] = p_val
        significance_row[col] = 'X' if p_val < 0.1 else ''

    stats_df.loc['P-Value'] = p_value_row
    stats_df.loc['Significance at 10%'] = significance_row

    # Append the statistical rows to the original matrix with an empty row in between
    matrix_with_stats = pd.concat([matrix, stats_df])

    return matrix_with_stats


# Perform two-sample t-tests between every set of variables in the average matrix
def compute_ttest_matrix(avg_matrix):
    avg_matrix = avg_matrix.apply(pd.to_numeric, errors='coerce')
    variables = avg_matrix.columns
    ttest_matrix = pd.DataFrame(index=variables, columns=variables)
    for var1 in variables:
        for var2 in variables:
            t_stat, p_val = stats.ttest_ind(avg_matrix[var1].dropna(), avg_matrix[var2].dropna())
            ttest_matrix.at[var1, var2] = p_val
    return ttest_matrix


# Perform PCA on the t-test matrix
def perform_pca(ttest_matrix):
    pca = PCA(n_components=min(ttest_matrix.shape[0], ttest_matrix.shape[1]))
    principal_components = pca.fit_transform(ttest_matrix)
    pca_df = pd.DataFrame(data=principal_components, index=ttest_matrix.index)
    explained_variance = pd.Series(pca.explained_variance_ratio_, name='Explained Variance')
    return pca_df, explained_variance


# Truncate sheet name to 31 characters and ensure uniqueness
def truncate_sheet_name(name, existing_names):
    truncated_name = name[:31]
    if truncated_name in existing_names:
        suffix = 1
        while f"{truncated_name[:28]}_{suffix}" in existing_names:
            suffix += 1
        truncated_name = f"{truncated_name[:28]}_{suffix}"
    existing_names.add(truncated_name)
    return truncated_name


# Main function
def main():

    # Load configuration
    config = load_config("config.ini")
    results_folder = config.get("Directories", "results_folder")
    results_file = config.get("Files", "school_points_fraction")
    school_n = config.getint("Processing", "SCHOOL_N")

    # Load the data
    file_path = os.path.join(results_folder, results_file)
    df = pd.read_excel(file_path)

    # Get school columns (robust to number of variables)
    school_columns = get_school_columns(df)[:school_n]

    # Get variable columns
    variable_columns = df.columns[:df.columns.get_loc(school_columns[0])]

    # Create a list of variable names based on the criteria and their indices
    variable_names = []
    for idx, row in df.iterrows():
        variable_name = get_variable_name(row, variable_columns)
        if not variable_name:
            variable_name = f"UnnamedVar{idx + 1}"
        variable_names.append(variable_name)

    # Add a column 'Variable' to the DataFrame to store these names
    df['Variable'] = variable_names

    # Create a list to hold all matrices for averaging
    matrices = []

    # Create Excel writer for individual school matrices
    output_file = os.path.join(results_folder, 'school_symmetrical_matrices.xlsx')
    existing_names = set()

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for school in school_columns:
            truncated_school_name = truncate_sheet_name(school, existing_names)

            # Create the symmetrical matrix
            matrix = create_symmetrical_matrix(df, school, variable_names)
            matrices.append(matrix)

            # Add statistics to the matrix
            matrix_with_stats = add_statistics_to_matrix(matrix, matrices)

            # Write the matrix with statistics to Excel
            matrix_with_stats.to_excel(writer, sheet_name=truncated_school_name)

    # Calculate the average matrix
    avg_matrix = pd.DataFrame(index=variable_names, columns=variable_names)
    for var1 in variable_names:
        for var2 in variable_names:
            avg_matrix.at[var1, var2] = sum(mtx.at[var1, var2] for mtx in matrices) / len(matrices)

    # Add a special bottom row with the sum of each column in the average matrix
    avg_matrix.loc['Total'] = avg_matrix.sum(axis=0)

    # Add statistics to the average matrix
    avg_matrix_with_stats = add_statistics_to_matrix(avg_matrix, matrices)

    # Compute the t-test matrix
    ttest_matrix = compute_ttest_matrix(avg_matrix)

    # Perform PCA on the t-test matrix
    pca_df, explained_variance = perform_pca(ttest_matrix)

    # Write the average matrix with statistics, the t-test matrix, and PCA results to separate Excel files
    average_output_file = os.path.join(results_folder, 'average_matrix.xlsx')
    ttest_output_file = os.path.join(results_folder, 'ttest_matrix.xlsx')
    pca_output_file = os.path.join(results_folder, 'pca_results.xlsx')
    with pd.ExcelWriter(average_output_file, engine='xlsxwriter') as writer:
        avg_matrix_with_stats.to_excel(writer, sheet_name='AVERAGE')
    with pd.ExcelWriter(ttest_output_file, engine='xlsxwriter') as writer:
        ttest_matrix.to_excel(writer, sheet_name='TTEST_P_VALUES')
    with pd.ExcelWriter(pca_output_file, engine='xlsxwriter') as writer:
        pca_df.to_excel(writer, sheet_name='PCA_Components')
        explained_variance.to_excel(writer, sheet_name='Explained_Variance')


if __name__ == "__main__":
    main()
