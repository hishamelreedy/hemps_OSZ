#ifndef AGENT_DEBUG_LOG_H
#define AGENT_DEBUG_LOG_H

#include <cstdio>
#include <ctime>

/* Session 44b7c9: append NDJSON to sim cwd (mlp784_clean/debug-44b7c9.log). */
static inline void agent_debug_log(
	const char *hypothesisId,
	const char *location,
	const char *message,
	unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4, unsigned int u5
) {
	static unsigned int agent_dbg_count;
	FILE *fp;

	if (agent_dbg_count >= 64u)
		return;
	agent_dbg_count++;

	fp = fopen("debug-44b7c9.log", "a");
	if (fp == NULL)
		fp = fopen("log/debug-44b7c9.log", "a");
	if (fp == NULL)
		return;
	fprintf(fp,
		"{\"sessionId\":\"44b7c9\",\"hypothesisId\":\"%s\",\"location\":\"%s\","
		"\"message\":\"%s\",\"data\":{\"u1\":%u,\"u2\":%u,\"u3\":%u,\"u4\":%u,\"u5\":%u},"
		"\"timestamp\":%lu}\n",
		hypothesisId, location, message,
		u1, u2, u3, u4, u5,
		(unsigned long)time(NULL) * 1000ul);
	fclose(fp);
}

#endif
