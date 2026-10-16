#ifndef BRAID_BRAID_HPP_
#define BRAID_BRAID_HPP_
#include <memory>
#include <stdexcept>
#include "braid/braid.h"
namespace braid {
struct code_deleter { void operator()(braid_code_t* p) const noexcept { braid_destroy_code(p); } };
struct plan_deleter { void operator()(braid_plan_t* p) const noexcept { braid_destroy_plan(p); } };
using code_ptr = std::unique_ptr<braid_code_t, code_deleter>;
using plan_ptr = std::unique_ptr<braid_plan_t, plan_deleter>;
inline braid_options_t default_options() {
  braid_options_t options;
  braid_options_init(&options, sizeof(options));
  return options;
}
inline code_ptr make_code(int k, int r, const braid_options_t* options = nullptr) {
  braid_code_t* result = nullptr;
  const auto status = braid_create_code_ex(k, r, options, options ? sizeof(*options) : 0, &result);
  if (status != BRAID_STATUS_OK) throw std::runtime_error(braid_status_string(status));
  return code_ptr(result);
}
}
#endif
