# ChatGPT for School Choice

This program automates the querying of the OpenAI ChatGPT API to collect and analyze information about school choice rankings provided by ChatGPT based on detailed prompts. It processes permutations of variable values from Excel files, queries the API, extracts and matches school names, and generates detailed reports and analyses.

In order to collect data at scale, this program uses a custom version of the ChatGPT API which I have implemented in C so that numerous queries can be made at once without being limited by Python's GIL. 
## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [Reading Templates and Variables](#reading-templates-and-variables)
  - [Generating Permutations](#generating-permutations)
  - [Prompt Substitution and API Query](#prompt-substitution-and-api-query)
  - [Extracting and Normalizing Names](#extracting-and-normalizing-names)
  - [Matching School Names](#matching-school-names)
  - [Processing Permutations](#processing-permutations)
  - [Writing Results to Excel](#writing-results-to-excel)
  - [Calculating Ordinal Results](#calculating-ordinal-results)
  - [Saving and Loading Intermediate Results](#saving-and-loading-intermediate-results)
  - [Configuration and Calculation of School Points](#configuration-and-calculation-of-school-points)
  - [Matrix Analysis Functions](#matrix-analysis-functions)
- [C and Cython Details](#c-and-cython-details)
- [Main Workflow](#main-workflow)
- [License](#license)

## Overview

This program is designed to help analyze how ChatGPT ranks schools based on different input variables. By querying the OpenAI API with different permutations of variables, the program can generate insights into how different factors influence the rankings provided by ChatGPT.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-repo/chatgpt-batch-query.git
    cd chatgpt-batch-query
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Install C dependencies:**

    **For Ubuntu/Debian:**
    ```sh
    sudo apt-get update
    sudo apt-get install libcurl4-openssl-dev build-essential python3-dev
    ```

    **For MacOS:**
    ```sh
    brew install curl
    ```

    **For Windows:**
    - Install Visual Studio with C++ development tools.

5. **Compile the Cython code:**

    ```sh
    python setup.py build_ext --inplace
    ```

## Configuration

The program uses a configuration file (`config.ini`) to manage settings. This file should be customized to suit your needs:

```ini
[API]
key = your_openai_api_key

[API_Limits]
max_tokens_per_minute = 160000
max_requests_per_minute = 5000
max_tokens_per_request = 300
max_retries = 3
retry_delay = 2

[Directories]
dependencies = path/to/dependencies
results_folder = path/to/results
temp_folder = path/to/temp

[Files]
prompt_template = prompt_template.txt
variables = variables.xlsx
school_list = school_list.xlsx

[Processing]
num_threads = 10
batch_size = 5
max_permutations = 1000
trials = 3
multi = true
timer = true
SCHOOL_N = 10
```

### Configuration Parameters

- **[API]**
  - `key`: Your OpenAI API key.

- **[API_Limits]**
  - `max_tokens_per_minute`: Maximum tokens allowed per minute by the API.
  - `max_requests_per_minute`: Maximum requests allowed per minute by the API.
  - `max_tokens_per_request`: Maximum tokens per individual request.
  - `max_retries`: Number of retries for failed requests.
  - `retry_delay`: Delay in seconds between retries.

- **[Directories]**
  - `dependencies`: Path to the folder containing dependencies.
  - `results_folder`: Path to the folder for storing result files.
  - `temp_folder`: Path to the folder for storing temporary files.

- **[Files]**
  - `prompt_template`: Name of the prompt template file.
  - `variables`: Name of the Excel file containing variables.
  - `school_list`: Name of the Excel file containing school names.

- **[Processing]**
  - `num_threads`: Number of threads to use for querying the API.
  - `batch_size`: Number of permutations to process in each batch.
  - `max_permutations`: Maximum number of permutations to process per iteration.
  - `trials`: Number of trials to run.
  - `multi`: Whether to generate all permutations (`true`) or selective permutations (`false`).
  - `timer`: Enable timer to measure execution time if `true`.
  - `SCHOOL_N`: Number of schools to consider in the matrix analysis.

## Usage

To run the data collection and analysis program, execute the following command:

```sh
python main.py
```

The `main.py` program will read the configuration, process permutations, query the OpenAI API, extract and match school names, and save the results to Excel files.

If you are interested in changing the prompt, simply modify `prompt_template.txt`. Words in brackets (such as `[income]`) in the prompt will be substituted by variables from `variables.xlsx`. To change the schools considered valid answers by the program, modify `school_list.xlsx`.

## How It Works

### Reading Templates and Variables

- **Purpose**: Load the prompt template and variable values from specified files.
- **Functions**:
  - `read_prompt_template(filename)`: Reads a prompt template from a text file.
  - `read_variable_values_from_excel(filename)`: Reads an Excel file containing variable values.

### Generating Permutations

- **Purpose**: Create different combinations of variable values to use as input for the ChatGPT API.
- **Functions**:
  - `generate_all_permutations(variables)`: Generates all possible permutations of variable values.
  - `generate_selective_permutations(variables, defaults)`: Generates selective permutations by varying one variable at a time while keeping others at their default values.

### Prompt Substitution and API Query

- **Purpose**: Substitute variable values into the prompt template and query the ChatGPT API.
- **Functions**:
  - `substitute_prompt(template, variables)`: Replaces placeholders in the prompt template with actual variable values.
  - `query_chatgpt_batch_with_retries(prompts, api_key, num_threads, temp_folder, batch_index, token_times, request_times, lock)`: Queries the ChatGPT API with the given prompts, handling retries and rate limits.

### Extracting and Normalizing Names

- **Purpose**: Extract and standardize school names from the API responses.
- **Functions**:
  - `extract_names_from_response(response)`: Extracts names from the API response using regular expressions.
  - `normalize_name(name)`: Normalizes school names by removing non-word characters.
  - `remove_excluded_words(name)`: Removes certain excluded words from the school names.

### Matching School Names

- **Purpose**: Match the extracted school names with a list of known school names to ensure consistency.
- **Functions**:
  - `match_school_names(extracted_names, school_names)`: Matches extracted names with a list of known school names based on common words.

### Processing Permutations

- **Purpose**: Process each permutation by substituting variables into the prompt, querying the API, and extracting/matching names.
- **Functions**:
  - `process_batch_of_permutations(batch, prompt_template, api_key, school_names, num_threads, temp_folder, batch_index, token_times, request_times, lock)`: Processes a batch of permutations, substitutes variables into the prompt, queries the API, extracts and matches names, and returns the results along with the prompts and responses.

### Writing Results to Excel

- **Purpose**: Save the results of the API queries to Excel files for further analysis.
- **Functions**:
  - `write_results_to_excel(results, filename)`: Writes the results to an Excel file.
  - `write_ordinal_results_to_excel(ordinal_results, filename)`: Writes ordinal results to a separate Excel file.

### Calculating Ordinal Results

- **Purpose**: Assign points to schools based on their rankings in different permutations.
- **Functions**:
  - `calculate_ordinal_results(results)`: Calculates ordinal results based on points assigned to schools in different permutations.

### Saving and Loading Intermediate Results

- **Purpose**: Save and load intermediate results to avoid repeating the entire process if something goes wrong.
- **Functions**:
  - `save_intermediate_results(results, queries, filename_prefix, batch_number)`: Saves intermediate results and queries to files.
  - `load_all_intermediate_results(filename_prefix)`: Loads all intermediate results from saved files.
  - `load_all_intermediate_queries(filename_prefix)`: Loads all intermediate queries from saved files.

### Configuration and Calculation of School Points

- **Purpose**: Calculate the overall contributions of each school based on the collected data.
- **Functions**:
  - `load_config(config_file)`: Loads configuration from a file.
  - `calculate_school_points(ordinal_results, output_folder)`: Calculates school points and contributions, saving the results to an Excel file.

### Matrix Analysis Functions

- **Purpose**: Perform detailed statistical analysis on the results.
- **Functions**:
  -

 `load_config(config_file)`: Loads configuration from a file.
  - `get_school_columns(df)`: Identifies columns in the dataframe that represent schools.
  - `is_numeric(s)`: Checks if a string is numeric.
  - `get_variable_name(row, variable_columns)`: Extracts variable names from rows based on specific criteria.
  - `create_symmetrical_matrix(df, school, variable_names)`: Creates a symmetrical matrix for each school based on variable names.
  - `add_statistics_to_matrix(matrix, matrices)`: Adds statistical data (mean, standard deviation, p-values) to matrices.
  - `compute_ttest_matrix(avg_matrix)`: Performs two-sample t-tests between every set of variables in the average matrix.
  - `perform_pca(ttest_matrix)`: Performs PCA on the t-test matrix.
  - `truncate_sheet_name(name, existing_names)`: Truncates sheet names to 31 characters and ensures uniqueness.

## C and Cython Details

### C Code (`chatgpt_query.c`)

The C code handles the multi-threaded querying of the ChatGPT API using cURL for HTTP requests. Here's a breakdown of its key functions:

1. **Includes and Defines**:
   - Includes necessary libraries for functionality.
   - Defines a buffer size for reading responses.

2. **Data Structure**:
   - Defines a structure (`thread_data_t`) to hold thread-specific data like thread ID, API key, prompt, and response.

3. **Callback Function**:
   - **`write_callback`**: Handles data received from the cURL request by appending it to a buffer.

4. **Escape Quotes Function**:
   - **`escape_quotes`**: Escapes quotes in the prompt string to format it correctly in JSON.

5. **Thread Function**:
   - **`query_chatgpt`**: Queries the ChatGPT API within a thread, using cURL to send HTTP requests and receive responses.

6. **Batch Initialization Function**:
   - **`init_query_batch`**: Initializes and manages a batch of ChatGPT queries, creating a thread for each query and collecting responses.

### Cython Code (`chatgpt_wrapper.pyx`)

The Cython code provides a Python interface to the C functions for querying the ChatGPT API in batches. Here's a breakdown of its key components:

1. **Imports**:
   - Imports necessary libraries from C and Python for memory management, string operations, file operations, and error printing.

2. **External C Function Declaration**:
   - Declares the external C function (`init_query_batch`) from `chatgpt_query.h`.

3. **Python Function**:
   - **`query_chatgpt_batch`**: Defines a Python function to query ChatGPT in batches, handling memory allocation, prompt encoding, and calling the external C function.

4. **Memory Management**:
   - Allocates and frees memory for API key, prompts, and responses.

5. **Error Handling**:
   - Includes error handling for memory allocation failures and exceptions during the API query process.

## Main Workflow

The `main` function orchestrates the entire workflow:

1. Loads configuration settings.
2. Reads necessary files (prompt template, variables, school list).
3. Generates permutations of variables.
4. Initializes folders for temporary and result files.
5. Processes permutations in batches using parallel processing.
6. Saves intermediate results and queries.
7. Loads all intermediate results and queries.
8. Calculates ordinal results and writes them to Excel files.
9. Calculates school points and writes the results to an Excel file.
10. Cleans up temporary files.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
