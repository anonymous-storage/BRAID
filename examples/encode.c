#include <braid/braid.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  uint8_t data[2][257], parity[257], restored[257];
  const uint8_t* inputs[] = {data[0], data[1]};
  uint8_t* outputs[] = {parity};
  const uint8_t* available[] = {NULL, data[1], parity};
  uint8_t* recovered[] = {restored};
  const int erased[] = {0};
  braid_options_t options;
  braid_code_t* code = NULL;
  if (braid_options_init(&options, sizeof(options)) != BRAID_STATUS_OK) return 1;
  options.num_threads = 1;
  if (braid_create_code_ex(2, 1, &options, sizeof(options), &code) != BRAID_STATUS_OK) return 2;
  for (size_t i = 0; i < sizeof(parity); ++i) {
    data[0][i] = (uint8_t)i;
    data[1][i] = (uint8_t)(i * 7);
  }
  int failed = braid_encode(code, inputs, outputs, sizeof(parity)) != BRAID_STATUS_OK;
  if (!failed) failed = braid_decode(code, available, erased, 1, recovered, sizeof(parity)) != BRAID_STATUS_OK;
  if (!failed) failed = memcmp(restored, data[0], sizeof(restored)) != 0;
  braid_destroy_code(code);
  if (!failed) printf("BRAID %s: C encode/decode OK\n", braid_version());
  return failed;
}
