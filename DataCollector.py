import itertools
import pandas as pd
import re
from openai import OpenAI
import xlsxwriter
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os
import pickle
import logging
import configparser

# Set up logging to file
logging.basicConfig(filename='error.log', level=logging.ERROR)

# Function to read the prompt template from the text file
def read_prompt_template(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

# Function to read the Excel file and get variable values
def read_variable_values_from_excel(filename):
    df = pd.read_excel(filename)
    variables = {col: df[col].dropna().astype(str).tolist() for col in df.columns}
    return variables

# Function to generate all possible permutations
def generate_all_permutations(variables):
    keys, values = zip(*variables.items())
    permutations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return permutations

# Function to generate selective permutations
def generate_selective_permutations(variables, defaults):
    permutations = []
    for key in variables.keys():
        for value in variables[key]:
            permutation = defaults.copy()
            permutation[key] = value
            permutations.append(permutation)
    return permutations

# Function to substitute variables into the prompt template
def substitute_prompt(template, variables):
    result_prompt = template
    for key, value in variables.items():
        if value == "None" or value.lower() == "exclude":
            result_prompt = result_prompt.replace(f"[{key}]", "")
        else:
            result_prompt = result_prompt.replace(f"[{key}]", value)
    result_prompt = re.sub(r'\s+', ' ', result_prompt).strip()
    return result_prompt

# Function to query the ChatGPT API
def query_chatgpt(prompt, client):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error querying ChatGPT API: {e}")
        return None

# Function to extract names from ChatGPT response
def extract_names_from_response(response):
    if response is None:
        return []
    pattern = r'\d+\.\s([^\n,]+)\s-\s'
    names = re.findall(pattern, response)
    return names[:5]

# Function to normalize school names
def normalize_name(name):
    return re.sub(r'[^\w\s]', '', name).strip()

# Function to remove excluded words
def remove_excluded_words(name):
    excluded_words = {"elementary", "alternative", "school", "academy"}
    words = name.lower().split()
    return ' '.join(word for word in words if word not in excluded_words)

# Function to match names by finding a common word
def match_school_names(extracted_names, school_names):
    matched_names = []
    normalized_school_names = [normalize_name(name) for name in school_names]
    modified_school_names = [remove_excluded_words(name) for name in normalized_school_names]

    for name in extracted_names:
        normalized_name = normalize_name(name)
        modified_name = remove_excluded_words(normalized_name)
        words_in_name = set(modified_name.split())

        found_match = False
        for school_name, modified_school_name in zip(school_names, modified_school_names):
            words_in_school_name = set(modified_school_name.split())
            if words_in_name & words_in_school_name:
                matched_names.append(school_name)
                found_match = True
                break

        if not found_match:
            matched_names.append("not_in_sfusd")

    return matched_names

# Function to process a single permutation
def process_permutation(variables, prompt_template, api_key, school_names):
    client = OpenAI(api_key=api_key)
    prompt = substitute_prompt(prompt_template, variables)
    response = query_chatgpt(prompt, client)
    names = extract_names_from_response(response)
    result = variables.copy()

    matched_names = match_school_names(names, school_names)

    for i, matched_name in enumerate(matched_names, start=1):
        result[f"Rank_{i}"] = matched_name

    if isinstance(result, dict):
        return (result, prompt, response)
    else:
        logging.error(f"Unexpected result type: {type(result)} - Value: {result}")
        return ({}, prompt, response)

# Function to write results to an Excel file
def write_results_to_excel(results, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                result = {k: str(v) for k, v in result.items()}
                df = pd.DataFrame([result])
            else:
                logging.error(f"Non-dict result at index {i}: {result}")
                df = pd.DataFrame([{"error": "Invalid result type"}])
            df.to_excel(writer, sheet_name=f"Results {i}", index=False)

# Function to write ordinal results to a separate Excel file
def write_ordinal_results_to_excel(ordinal_results, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df = pd.DataFrame(ordinal_results)
        df.to_excel(writer, sheet_name="Ordinal Results", index=False)

# Function to calculate ordinal results based on points
def calculate_ordinal_results(results):
    permutation_points = defaultdict(lambda: defaultdict(int))

    for trial_results in results:
        for result in trial_results:
            if isinstance(result, dict):
                permutation = {key: value for key, value in result.items() if not key.startswith('Rank_')}
                for rank in range(1, 6):
                    school = result.get(f"Rank_{rank}")
                    if school:
                        points = 6 - rank
                        permutation_points[tuple(permutation.items())][school] += points
            else:
                logging.error(f"Unexpected result type in trial results: {result}")

    ordinal_results = []
    for permutation, school_points in permutation_points.items():
        sorted_schools = sorted(school_points.items(), key=lambda x: x[1], reverse=True)
        result = dict(permutation)
        result.update({f"Rank_{i + 1}": school for i, (school, _) in enumerate(sorted_schools)})
        ordinal_results.append(result)
    return ordinal_results

# Function to save intermediate results
def save_intermediate_results(results, filename_prefix, batch_number):
    with open(f"{filename_prefix}/batch_{batch_number}.pkl", 'wb') as f:
        pickle.dump(results, f)

# Function to load all intermediate results
def load_all_intermediate_results(filename_prefix, num_batches):
    all_results = []
    for batch_number in range(num_batches):
        with open(f"{filename_prefix}/batch_{batch_number}.pkl", 'rb') as f:
            batch_results = pickle.load(f)
            all_results.extend(batch_results)
    return all_results

# Function to save intermediate queries
def save_intermediate_queries(queries, filename_prefix, batch_number):
    with open(f"{filename_prefix}/batch_{batch_number}.pkl", 'wb') as f:
        pickle.dump(queries, f)

# Function to load all intermediate queries
def load_all_intermediate_queries(filename_prefix, num_batches):
    all_queries = []
    for batch_number in range(num_batches):
        with open(f"{filename_prefix}/batch_{batch_number}.pkl", 'rb') as f:
            batch_queries = pickle.load(f)
            all_queries.extend(batch_queries)
    return all_queries

# Function to load configuration from file
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

# Function to calculate school points and contributions
def calculate_school_points(file_path, output_folder):
    school_points = {}
    row_contributions = {}

    xls = pd.ExcelFile(file_path)
    max_rows = 0
    variables = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        max_rows = max(max_rows, len(df))

        school_columns = [col for col in df.columns if col.startswith('Rank_')]
        variable_columns = [col for col in df.columns if not col.startswith('Rank_')]
        points_distribution = list(range(len(school_columns), 0, -1))

        for index, row in df.iterrows():
            variables[index] = row[variable_columns].to_dict()
            for col_index, points in zip(school_columns, points_distribution):
                if col_index in df.columns:
                    school = row[col_index]
                    if pd.notna(school):
                        if school not in school_points:
                            school_points[school] = 0
                        school_points[school] += points

                        row_key = index + 1
                        if school not in row_contributions:
                            row_contributions[school] = {}
                        if row_key not in row_contributions[school]:
                            row_contributions[school][row_key] = 0
                        row_contributions[school][row_key] += points

    top_schools = dict(sorted(school_points.items(), key=lambda item: item[1], reverse=True))
    result_data = []
    for school in top_schools:
        total_points = top_schools[school]
        row_contributions_fraction = {}
        for row_key, points in row_contributions[school].items():
            fraction = points / total_points
            row_contributions_fraction[row_key] = fraction
        result_data.append(row_contributions_fraction)

    rows = range(1, max_rows + 1)
    result_df = pd.DataFrame(index=rows, columns=["Row"] + variable_columns + list(top_schools.keys()))

    for row in rows:
        result_df.at[row, "Row"] = row
        for var in variable_columns:
            result_df.at[row, var] = variables.get(row - 1, {}).get(var, "")
        for school, row_data in zip(top_schools.keys(), result_data):
            result_df.at[row, school] = row_data.get(row, 0)

    output_file = os.path.join(output_folder, 'school_points_fraction.xlsx')
    result_df.to_excel(output_file, index=False)

    return result_df

# Main function
def main():
    config = load_config("config.ini")
    api_key = config.get("API", "key")
    dependencies_folder = config.get("Directories", "dependencies")

    prompt_template_path = os.path.join(dependencies_folder, config.get("Files", "prompt_template"))
    variables_path = os.path.join(dependencies_folder, config.get("Files", "variables"))
    school_list_path = os.path.join(dependencies_folder, config.get("Files", "school_list"))

    prompt_template = read_prompt_template(prompt_template_path)
    variables = read_variable_values_from_excel(variables_path)

    multi = config.getboolean("Processing", "multi")
    if multi:
        permutations = generate_all_permutations(variables)
    else:
        defaults = {
            'ethnicity': 'None',
            'income': 'family',
            'child ability': 'None',
            'child gender': 'kid',
            'child education level': 'K level',
            'parent education': 'None',
            'location': 'None'
        }
        permutations = generate_selective_permutations(variables, defaults)

    school_list_df = pd.read_excel(school_list_path)
    school_names = school_list_df['school_name_long'].dropna().astype(str).tolist()

    batch_size = config.getint("Processing", "batch_size")
    max_permutations = config.getint("Processing", "max_permutations")
    trials = config.getint("Processing", "trials")
    temp_folder = config.get("Directories", "temp_folder")
    results_folder = config.get("Directories", "results_folder")

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    permutations = permutations[:max_permutations]
    print(len(permutations))

    results = []
    all_queries = []
    batch_number = 0
    query_batch_number = 0

    with Pool(cpu_count()) as pool:
        for trial in range(trials):
            print(f"Trial {trial + 1}")
            for i in range(0, len(permutations), batch_size):
                batch_permutations = permutations[i:i + batch_size]
                batch_data = [(perm, prompt_template, api_key, school_names) for perm in batch_permutations]
                batch_results = pool.starmap(process_permutation, batch_data)
                valid_results = [(res, prompt, response) for res, prompt, response in batch_results if
                                 isinstance(res, dict) and res]
                if valid_results:
                    results.extend(valid_results)
                    all_queries.extend(valid_results)
                    save_intermediate_results([(res, prompt) for res, prompt, _ in valid_results], temp_folder, batch_number)
                    save_intermediate_queries([(res, prompt, response) for res, prompt, response in valid_results],
                                              temp_folder, query_batch_number)
                    batch_number += 1
                    query_batch_number += 1
                    results = []
                    all_queries = []

    final_results = load_all_intermediate_results(temp_folder, batch_number)
    all_queries = load_all_intermediate_queries(temp_folder, query_batch_number)
    ordinal_results = calculate_ordinal_results(final_results)

    for i, result in enumerate(final_results):
        if not isinstance(result, dict):
            logging.error(f"Unexpected result type at index {i}: {type(result)} - Value: {result}")

    with open(f"{results_folder}/all_queries.txt", 'w', encoding='utf-8') as f:
        for result, prompt, response in all_queries:
            f.write(f"Prompt: {prompt}\nResponse: {result}\nFull Response: {response}\n\n")

    ordinal_results_file = f"{results_folder}/ordinal_results.xlsx"
    write_ordinal_results_to_excel(ordinal_results, ordinal_results_file)

    write_results_to_excel(final_results, f"{results_folder}/results.xlsx")

    calculate_school_points(ordinal_results_file, results_folder)

    for i in range(batch_number):
        file_path = f"{temp_folder}/batch_{i}.pkl"
        if os.path.exists(file_path):
            os.remove(file_path)

    for i in range(query_batch_number):
        file_path = f"{temp_folder}/batch_{i}.pkl"
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    main()