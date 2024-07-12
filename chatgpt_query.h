#ifndef CHATGPT_QUERY_H
#define CHATGPT_QUERY_H

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

void init_query(const char *api_key, const char *prompt, int num_threads, char **responses);

#ifdef __cplusplus
}
#endif

#endif // CHATGPT_QUERY_H
