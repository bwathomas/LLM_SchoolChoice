# ChatGPT for School Choice

This program automates the querying of the OpenAI ChatGPT API to collect and analyze information about the kind of school choice rankings ChatGPT provides based on prompt details.
The program reads templates and variable values from Excel files, processes permutations of these variables, queries the API, extracts and matches school names, and generates detailed reports and analyses.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Functions](#functions)
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
- [Main Workflow](#main-workflow)
- [License](#license)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/bwathomas/LLM_SchoolChoice.git
    cd LLM_SchoolChoice
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Configuration

The program uses a configuration file (`config.ini`) to manage settings. The configuration file should have the following structure:

```ini
[API]
key = your_openai_api_key

[Directories]
dependencies = path/to/dependencies
temp_folder = path/to/temp
results_folder = path/to/results

[Files]
prompt_template = prompt_template.txt
variables = variables.xlsx
school_list = school_list.xlsx

[Processing]
multi = true
batch_size = 100
max_permutations = 1000
trials = 3
```

- `key`: Your OpenAI API key.
- `dependencies`: Path to the folder containing dependencies.
- `temp_folder`: Path to the folder for storing temporary files.
- `results_folder`: Path to the folder for storing result files.
- `prompt_template`: Name of the prompt template file.
- `variables`: Name of the Excel file containing variables.
- `school_list`: Name of the Excel file containing school names.
- `multi`: Whether to generate all permutations (`true`) or selective permutations (`false`).
- `batch_size`: Number of permutations to process in each batch (this prevents memory issues when large numbers of queries are required).
- `max_permutations`: Maximum number of permutations to process per iteration (this is primarily for testing).
- `trials`: Number of trials to run.

## Usage

To run the program, execute the following command:

```sh
python main.py
```

The program will read the configuration, process permutations, query the OpenAI API, extract and match school names, and save the results to Excel files.

In the event you are interested in changing the prompt, simply change prompt.txt. Words in bracket (such as [income]) in the prompt will be substituted by variables from Variables.xlsx. In order to change the schools considered valid answers by the program, change school_list.xlsx.
## Functions

### Reading Templates and Variables

- `read_prompt_template(filename)`: Reads a prompt template from a text file.
- `read_variable_values_from_excel(filename)`: Reads an Excel file containing variable values.

### Generating Permutations

- `generate_all_permutations(variables)`: Generates all possible permutations of variable values.
- `generate_selective_permutations(variables, defaults)`: Generates selective permutations by varying one variable at a time while keeping others at their default values.

### Prompt Substitution and API Query

- `substitute_prompt(template, variables)`: Replaces placeholders in the prompt template with actual variable values.
- `query_chatgpt(prompt, client)`: Queries the ChatGPT API with the given prompt and returns the response.

### Extracting and Normalizing Names

- `extract_names_from_response(response)`: Extracts names from the API response using regular expressions.
- `normalize_name(name)`: Normalizes school names by removing non-word characters.
- `remove_excluded_words(name)`: Removes certain excluded words from the school names.

### Matching School Names

- `match_school_names(extracted_names, school_names)`: Matches extracted names with a list of known school names based on common words.

### Processing Permutations

- `process_permutation(variables, prompt_template, api_key, school_names)`: Processes a single permutation, substitutes variables into the prompt, queries the API, extracts and matches names, and returns the result along with the prompt and response.

### Writing Results to Excel

- `write_results_to_excel(results, filename)`: Writes the results to an Excel file, with each result on a separate sheet.
- `write_ordinal_results_to_excel(ordinal_results, filename)`: Writes ordinal results to a separate Excel file.

### Calculating Ordinal Results

- `calculate_ordinal_results(results)`: Calculates ordinal results based on points assigned to schools in different permutations.

### Saving and Loading Intermediate Results

- `save_intermediate_results(results, filename_prefix, batch_number)`: Saves intermediate results to a file.
- `load_all_intermediate_results(filename_prefix, num_batches)`: Loads all intermediate results from saved files.
- `save_intermediate_queries(queries, filename_prefix, batch_number)`: Saves intermediate queries to a file.
- `load_all_intermediate_queries(filename_prefix, num_batches)`: Loads all intermediate queries from saved files.

### Configuration and Calculation of School Points

- `load_config(config_file)`: Loads configuration from a file.
- `calculate_school_points(file_path, output_folder)`: Calculates school points and contributions, saving the results to an Excel file.

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