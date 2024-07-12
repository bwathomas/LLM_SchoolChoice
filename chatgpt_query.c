// Include necessary header files for functionality
#include "chatgpt_query.h" // Custom header file for ChatGPT queries
#include <stdio.h> // Standard input/output library
#include <stdlib.h> // Standard library for memory allocation, process control, etc.
#include <string.h> // String manipulation functions
#include <curl/curl.h> // cURL library for making HTTP requests
#include <windows.h> // Windows API for threading

// Define the buffer size for reading responses
#define BUFFER_SIZE 2048

// Define a structure to hold thread-specific data
typedef struct {
    int thread_id; // Unique thread identifier
    char *api_key; // API key for authentication
    char *prompt; // User prompt for the ChatGPT query
    char *response; // Response from the ChatGPT API
} thread_data_t;

// Callback function to handle data received from the cURL request
size_t write_callback(void *ptr, size_t size, size_t nmemb, void *data) {
    // Append received data to the buffer
    strncat((char *)data, (char *)ptr, size * nmemb);
    return size * nmemb;
}

// Function to escape quotes in the prompt string
char* escape_quotes(const char *str) {
    int len = strlen(str); // Get the length of the input string
    char *escaped_str = (char *)malloc(len * 2 + 1); // Allocate memory for the escaped string (worst case)
    if (escaped_str == NULL) { // Check if memory allocation failed
        return NULL;
    }
    char *p = escaped_str; // Pointer to iterate over the escaped string
    for (int i = 0; i < len; i++) { // Iterate over the input string
        if (str[i] == '"') { // If the character is a quote
            *p++ = '\\'; // Add a backslash before the quote
        }
        *p++ = str[i]; // Copy the character
    }
    *p = '\0'; // Null-terminate the escaped string
    return escaped_str;
}

// Thread function to query the ChatGPT API
DWORD WINAPI query_chatgpt(LPVOID threadarg) {
    thread_data_t *data = (thread_data_t *)threadarg; // Retrieve thread-specific data
    CURL *curl;
    CURLcode res;
    char buffer[BUFFER_SIZE] = {0}; // Buffer to hold the response
    char postfields[BUFFER_SIZE]; // Buffer to hold the POST fields
    char auth_header[BUFFER_SIZE]; // Buffer to hold the authorization header

    // Escape quotes in the prompt to format it correctly in JSON
    char *escaped_prompt = escape_quotes(data->prompt);
    if (escaped_prompt == NULL) { // Check if memory allocation failed
        fprintf(stderr, "Failed to allocate memory for escaped prompt\n");
        return 1;
    }

    // Format the POST fields and authorization header
    snprintf(postfields, BUFFER_SIZE, "{\"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"user\", \"content\": \"%s\"}]}", escaped_prompt);
    snprintf(auth_header, BUFFER_SIZE, "Authorization: Bearer %s", data->api_key);

    curl = curl_easy_init(); // Initialize a cURL session

    if(curl) { // If cURL session initialized successfully
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json"); // Add content-type header
        headers = curl_slist_append(headers, auth_header); // Add authorization header

        // Set cURL options for the request
        curl_easy_setopt(curl, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postfields);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, buffer);

        res = curl_easy_perform(curl); // Perform the cURL request

        if(res != CURLE_OK) { // Check if the request failed
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            data->response = strdup(buffer); // Store the response in the thread data
        }

        curl_slist_free_all(headers); // Clean up headers
        curl_easy_cleanup(curl); // Clean up the cURL session
    }

    free(escaped_prompt); // Free the allocated memory for the escaped prompt
    return 0;
}

// Function to initialize and manage a batch of ChatGPT queries
void init_query_batch(const char *api_key, const char **prompts, int num_prompts, char **responses) {
    // Allocate memory for thread handles and thread-specific data
    HANDLE *threads = (HANDLE *)malloc(num_prompts * sizeof(HANDLE));
    thread_data_t *thread_data = (thread_data_t *)malloc(num_prompts * sizeof(thread_data_t));

    for(int i = 0; i < num_prompts; i++) {
        // Initialize thread-specific data
        thread_data[i].thread_id = i;
        thread_data[i].api_key = strdup(api_key);
        thread_data[i].prompt = strdup(prompts[i]);
        thread_data[i].response = NULL;

        // Create a new thread to handle the ChatGPT query
        threads[i] = CreateThread(NULL, 0, query_chatgpt, &thread_data[i], 0, NULL);
        if (threads[i] == NULL) { // Check if thread creation failed
            fprintf(stderr, "Error: unable to create thread %d\n", i);
            exit(1);
        }
    }

    // Wait for all threads to complete
    WaitForMultipleObjects(num_prompts, threads, TRUE, INFINITE);

    for(int i = 0; i < num_prompts; i++) {
        CloseHandle(threads[i]); // Close the thread handle
        responses[i] = thread_data[i].response; // Collect the responses
        free(thread_data[i].api_key); // Free allocated memory for API key
        free(thread_data[i].prompt); // Free allocated memory for prompt
    }

    free(threads); // Free allocated memory for thread handles
    free(thread_data); // Free allocated memory for thread-specific data
}
