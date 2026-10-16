#ifndef BRAID_EXPORT_H_
#define BRAID_EXPORT_H_
#if defined(__GNUC__) || defined(__clang__)
#define BRAID_API __attribute__((visibility("default")))
#else
#define BRAID_API
#endif
#endif
