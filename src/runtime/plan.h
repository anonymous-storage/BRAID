#ifndef BRAID_RUNTIME_PLAN_H_
#define BRAID_RUNTIME_PLAN_H_

#include <memory>
#include <vector>

#include "src/runtime/context.h"

namespace braid::runtime {

std::shared_ptr<Plan> BuildEncodePlan(Context& context, size_t block_size);
std::shared_ptr<Plan> BuildDecodePlan(Context& context, const std::vector<int>& erased_indices, size_t block_size);
std::shared_ptr<Plan> PrepareInterUnitBatchPlan(
    const Context& context, const std::shared_ptr<Plan>& base_plan);

}  // namespace braid::runtime

#endif  // BRAID_RUNTIME_PLAN_H_
