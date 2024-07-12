# Import necessary libraries from C and Python
from libc.stdlib cimport malloc, free  # Import malloc and free for memory management
from libc.string cimport strcpy, strlen  # Import strcpy and strlen for string operations
import os  # Import os module for file operations
import sys  # Import sys module for error printing
from cython.parallel import prange  # Import prange for parallel operations (not used here but available)

# Declare external C function from chatgpt_query.h
cdef extern from "chatgpt_query.h":
    void init_query_batch(const char * api_key, const char ** prompts, int num_prompts, char ** responses) nogil  # Declare external function

# Define a Python function to query ChatGPT in batches
def query_chatgpt_batch(list prompts, str api_key, int num_prompts, str temp_folder, int batch_index) -> list:
    cdef bytes api_key_bytes  # Byte-encoded API key
    cdef const char * c_api_key  # C string for API key
    cdef const char ** c_prompts  # C array of prompts
    cdef char ** responses  # C array of responses
    cdef int i  # Loop counter
    cdef list file_paths = []  # List to store file paths of responses

    # Encode API key to bytes and allocate memory for it
    api_key_bytes = api_key.encode('utf-8')
    c_api_key = <const char *> malloc(len(api_key_bytes) + 1)
    if c_api_key == NULL:
        raise MemoryError("Failed to allocate memory for API key")  # Handle memory allocation failure

    # Allocate memory for prompts
    c_prompts = <const char **> malloc(num_prompts * sizeof(char *))
    if c_prompts == NULL:
        free(<void *> c_api_key)
        raise MemoryError("Failed to allocate memory for prompts")  # Handle memory allocation failure

    # Encode each prompt and allocate memory for it
    for i in range(num_prompts):
        prompt_bytes = prompts[i].encode('utf-8')
        c_prompts[i] = <const char *> malloc(len(prompt_bytes) + 1)
        if c_prompts[i] == NULL:
            free(<void *> c_api_key)
            for j in range(i):
                free(<void *> c_prompts[j])
            free(<void *> c_prompts)
            raise MemoryError(f"Failed to allocate memory for prompt {i}")
        strcpy(<char *> c_prompts[i], prompt_bytes)  # Copy prompt to allocated memory

    # Allocate memory for responses
    responses = <char **> malloc(num_prompts * sizeof(char *))
    if responses == NULL:
        free(<void *> c_api_key)
        for i in range(num_prompts):
            free(<void *> c_prompts[i])
        free(<void *> c_prompts)
        raise MemoryError("Failed to allocate memory for responses")  # Handle memory allocation failure

    # Allocate memory for each response and ensure they are null-terminated
    for i in range(num_prompts):
        responses[i] = <char *> malloc(1024 * sizeof(char))
        if responses[i] == NULL:
            free(<void *> c_api_key)
            for j in range(num_prompts):
                free(<void *> c_prompts[j])
            free(<void *> c_prompts)
            for j in range(i):
                free(<void *> responses[j])
            free(<void *> responses)
            raise MemoryError(f"Failed to allocate memory for response {i}")
        responses[i][0] = '\0'  # Ensure the string is null-terminated

    try:
        # Copy API key to allocated memory
        strcpy(<char *> c_api_key, api_key_bytes)
        with nogil:  # Release GIL for calling external function
            init_query_batch(c_api_key, c_prompts, num_prompts, responses)

        # Write each response to a temporary file
        # Why not just return the result? This causes issue on Windows OS for unknown reasons
        for i in range(num_prompts):
            if responses[i] != NULL and responses[i][0] != '\0':
                file_path = os.path.join(temp_folder, f"response_{batch_index}_{i}.txt")
                file_paths.append(file_path)
                with open(file_path, "w") as f:
                    f.write(responses[i].decode('utf-8'))

        return file_paths  # Return list of file paths

    except Exception as e:
        print(f"Error in query_chatgpt_batch: {e}", file=sys.stderr)  # Print error message
        return []
    finally:
        # Free allocated memory
        free(<void *> c_api_key)
        for i in range(num_prompts):
            free(<void *> c_prompts[i])
        free(<void *> c_prompts)
        for i in range(num_prompts):
            if responses[i] != NULL:
                free(responses[i])
        free(<void *> responses)
