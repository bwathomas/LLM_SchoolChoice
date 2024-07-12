import itertools
import pandas as pd
import re
import xlsxwriter
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager, Lock
import os
import pickle
import logging
import configparser
import time
from datetime import datetime, timedelta

# Import the Cython wrapper
import chatgpt_wrapper

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

# Function to query the ChatGPT API with retries
def query_chatgpt_batch_with_retries(prompts, api_key, num_threads, temp_folder, batch_index, token_times, request_times, lock):
    results = []
    for attempt in range(MAX_RETRIES):
        with lock:
            current_time = datetime.now()

            # Remove tokens and requests older than a minute
            token_times[:] = [t for t in token_times if t > current_time - timedelta(minutes=1)]
            request_times[:] = [t for t in request_times if t > current_time - timedelta(minutes=1)]

            required_tokens = len(prompts) * MAX_TOKENS_PER_REQUEST

            # Convert ListProxy to a regular list for concatenation
            token_times_list = list(token_times)
            request_times_list = list(request_times)

            # Check if the current batch will exceed the limits
            if len(token_times_list) + required_tokens > MAX_TOKENS_PER_MINUTE or len(request_times_list) + num_threads > MAX_REQUESTS_PER_MINUTE:
                delay = min((t + timedelta(minutes=1) - current_time).total_seconds() for t in (token_times_list + request_times_list))
                logging.info(f"Delaying {delay} seconds to stay within rate limits")
                time.sleep(delay)
                continue  # Recheck limits after waiting

        try:
            batch_results = chatgpt_wrapper.query_chatgpt_batch(prompts, api_key, num_threads, temp_folder, batch_index)
            results.extend(batch_results)
            with lock:
                request_times.extend([current_time] * len(prompts))
                token_times.extend([current_time] * required_tokens)
            break
        except Exception as e:
            logging.error(f"Error querying ChatGPT API: {e}. Retry {attempt + 1} of {MAX_RETRIES}")
            time.sleep(RETRY_DELAY)
    else:
        logging.error(f"Max retries reached for batch {batch_index}. Adding empty responses.")
        results.extend([{} for _ in prompts])

    return results

# Function to extract school names from the response
def extract_names_from_response(response):
    if response is None:
        return []
    pattern = r'\d+\.\s*([^\n-]+?(?:\s(?:K-8|PreK-8|K-5|K-12))?(?:\s(?:School|Community School|Alternative School|Academy|Magnet School))?)\s*-\s'
    names = re.findall(pattern, response)
    return names[:5]

# Function to normalize school names
def normalize_name(name):
    return re.sub(r'[^\w\s]', '', name).strip()

# Function to remove excluded words from school names
def remove_excluded_words(name):
    excluded_words = {"elementary", "alternative", "school", "academy"}
    words = name.lower().split()
    return ' '.join(word for word in words if word not in excluded_words)

# Function to match extracted names with known school names
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

# Function to process a batch of permutations
def process_batch_of_permutations(batch, prompt_template, api_key, school_names, num_threads, temp_folder, batch_index, token_times, request_times, lock):
    prompts = [substitute_prompt(prompt_template, variables) for variables in batch]
    try:
        file_paths = query_chatgpt_batch_with_retries(prompts, api_key, num_threads, temp_folder, batch_index, token_times, request_times, lock)
    except IndexError as e:
        logging.error(f"IndexError in query_chatgpt_batch: {e}")
        file_paths = [None] * len(prompts)  # Ensure file_paths list is the same length as prompts

    results = []
    queries = []

    for i, file_path in enumerate(file_paths):
        if file_path:
            with open(file_path, "r") as f:
                response = f.read()
            os.remove(file_path)
            combined_response = response
            names = extract_names_from_response(combined_response)
            result = batch[i].copy()
            matched_names = match_school_names(names, school_names)
            for j, matched_name in enumerate(matched_names, start=1):
                result[f"Rank_{j}"] = matched_name
            results.append((result, prompts[i], combined_response))
            queries.append((result, prompts[i], combined_response))
        else:
            results.append(({}, prompts[i], ""))  # Ensure an empty response is recorded
            queries.append(({}, prompts[i], ""))  # Ensure an empty response is recorded

    return results, queries

# Function to write results to an Excel file
def write_results_to_excel(results, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df = pd.DataFrame([res[0] for res in results if isinstance(res[0], dict)])
        df.to_excel(writer, sheet_name="Results", index=False)

# Function to write ordinal results to a separate Excel file
def write_ordinal_results_to_excel(ordinal_results, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df = pd.DataFrame(ordinal_results)
        df.to_excel(writer, sheet_name="Ordinal Results", index=False)

# Function to calculate ordinal results based on points
def calculate_ordinal_results(results):
    permutation_points = defaultdict(lambda: defaultdict(int))

    for result, prompt, response in results:
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
def save_intermediate_results(results, queries, filename_prefix, batch_number):
    with open(f"{filename_prefix}/batch_{batch_number}_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    with open(f"{filename_prefix}/batch_{batch_number}_queries.pkl", 'wb') as f:
        pickle.dump(queries, f)

# Function to load all intermediate results
def load_all_intermediate_results(filename_prefix):
    all_results = []
    batch_files = [f for f in os.listdir(filename_prefix) if f.startswith('batch_') and f.endswith('_results.pkl')]
    for batch_file in batch_files:
        with open(os.path.join(filename_prefix, batch_file), 'rb') as f:
            batch_results = pickle.load(f)
            all_results.extend(batch_results)
        os.remove(os.path.join(filename_prefix, batch_file))
    return all_results

# Function to load all intermediate queries
def load_all_intermediate_queries(filename_prefix):
    all_queries = []
    batch_files = [f for f in os.listdir(filename_prefix) if f.startswith('batch_') and f.endswith('_queries.pkl')]
    for batch_file in batch_files:
        with open(os.path.join(filename_prefix, batch_file), 'rb') as f:
            batch_queries = pickle.load(f)
            all_queries.extend(batch_queries)
        os.remove(os.path.join(filename_prefix, batch_file))
    return all_queries

# Function to calculate school points and contributions
def calculate_school_points(ordinal_results, output_folder):
    school_points = {}
    row_contributions = {}

    max_rows = 0
    variables = {}

    # Collect all results into a single DataFrame
    df = pd.DataFrame(ordinal_results)

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

    rows = range(1, len(df) + 1)
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

# Function to load configuration from file
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

# Function to process permutations in parallel
def parallel_process_batch(args):
    return process_batch_of_permutations(*args)

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

    num_threads = config.getint("Processing", "num_threads")
    num_requests = num_threads
    batch_size = config.getint("Processing", "batch_size")

    # Load API limit constants from the config file
    global MAX_TOKENS_PER_MINUTE
    global MAX_REQUESTS_PER_MINUTE
    global MAX_TOKENS_PER_REQUEST
    global MAX_RETRIES
    global RETRY_DELAY
    MAX_TOKENS_PER_MINUTE = config.getint("API_Limits", "max_tokens_per_minute")
    MAX_REQUESTS_PER_MINUTE = config.getint("API_Limits", "max_requests_per_minute")
    MAX_TOKENS_PER_REQUEST = config.getint("API_Limits", "max_tokens_per_request")
    MAX_RETRIES = config.getint("API_Limits", "max_retries")
    RETRY_DELAY = config.getint("API_Limits", "retry_delay")

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

    max_permutations = config.getint("Processing", "max_permutations")
    results_folder = config.get("Directories", "results_folder")
    temp_folder = config.get("Directories", "temp_folder")
    trials = config.getint("Processing", "trials")
    timer = config.getboolean("Processing", "timer")

    if timer:
        start_time = time.time()

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    permutations = permutations[:max_permutations]

    with Manager() as manager:
        token_times = manager.list()
        request_times = manager.list()
        lock = manager.Lock()

        all_results = []
        all_queries = []
        batch_counter = 0

        for trial in range(trials):
            print(f"Starting trial {trial + 1} of {trials}")
            results = []
            queries = []
            batch_number = 0

            with Pool(cpu_count()) as pool:
                batch_args = [
                    (permutations[i:i + num_requests], prompt_template, api_key, school_names, num_threads, temp_folder, batch_number + i // num_requests, token_times, request_times, lock)
                    for i in range(0, len(permutations), num_requests)
                ]

                # Add logging to verify batch arguments
                logging.info(f"Batch arguments: {batch_args}")

                for i in range(0, len(batch_args), batch_size):
                    all_batch_results = pool.map(parallel_process_batch, batch_args[i:i + batch_size])

                    for batch_result, batch_query in all_batch_results:
                        results.extend(batch_result)
                        queries.extend(batch_query)
                        save_intermediate_results(batch_result, batch_query, temp_folder, batch_counter)
                        batch_counter += 1

                    results.clear()  # Clear RAM
                    queries.clear()  # Clear RAM

        # Load and delete all intermediate results and queries
        all_results = load_all_intermediate_results(temp_folder)
        all_queries = load_all_intermediate_queries(temp_folder)

        # Writing unified results
        with pd.ExcelWriter(f"{results_folder}/results.xlsx", engine='xlsxwriter') as writer:
            for trial in range(trials):
                trial_results = all_results[trial::trials]
                df = pd.DataFrame([res[0] for res in trial_results if isinstance(res[0], dict)])
                df.to_excel(writer, sheet_name=f"Results_Trial_{trial + 1}", index=False)

        ordinal_results = calculate_ordinal_results(all_results)
        write_ordinal_results_to_excel(ordinal_results, f"{results_folder}/ordinal_results.xlsx")

        calculate_school_points(ordinal_results, results_folder)

        with open(f"{results_folder}/all_queries.txt", 'w', encoding='utf-8') as f:
            for result, prompt, response in all_queries:
                f.write(f"Prompt: {prompt}\nResponse: {result}\nFull Response: {response}\n\n")

        for i in range(batch_counter):
            result_file_path = f"{temp_folder}/batch_{i}_results.pkl"
            query_file_path = f"{temp_folder}/batch_{i}_queries.pkl"
            if os.path.exists(result_file_path):
                os.remove(result_file_path)
            if os.path.exists(query_file_path):
                os.remove(query_file_path)

    if timer:
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
