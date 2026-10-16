#ifndef BRAID_RUNTIME_CONTEXT_H_
#define BRAID_RUNTIME_CONTEXT_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/decomposition.h"
#include "src/core/rs_matrix.h"
#include "src/runtime/topology.h"
#include "src/runtime/thread_pool.h"
#include "src/runtime/types.h"

namespace braid::runtime {

class Context {
 public:
  Context(int k, int r, braid_options_t options);

  int k() const { return k_; }
  int r() const { return r_; }
  int n() const { return n_; }
  const braid_options_t& options() const { return options_; }
  const BackendCaps& caps() const { return caps_; }
  const core::DecompositionTable& table() const { return table_; }
  const core::Matrix& generator() const { return generator_; }
  const Topology& topology() const { return topology_; }
  ThreadPool* thread_pool() const { return thread_pool_.get(); }
  bool pin_threads() const { return options_.pin_threads != 0; }
  bool numa_aware() const { return options_.numa_aware != 0; }

  braid_backend_t ResolveBackend() const;
  braid_path_t ResolvePath(PlanType type, int outputs, size_t block_size, braid_backend_t backend) const;
  std::pair<size_t, int> ChooseTileAndBatch(
      PlanType type,
      int inputs,
      int outputs,
      size_t block_size,
      braid_backend_t backend,
      braid_path_t path) const;

  std::shared_ptr<Plan> GetOrBuildEncodePlan(size_t block_size);
  std::shared_ptr<Plan> GetOrBuildDecodePlan(const std::vector<int>& erased_indices, size_t block_size);
  std::shared_ptr<Plan> GetOrBuildEncodeBatchPlan(size_t block_size);
  std::shared_ptr<Plan> GetOrBuildDecodeBatchPlan(
      const std::vector<int>& erased_indices, size_t block_size);
  int repr_generator() const { return table_.generator; }
  core::DecompositionTable SelectTableForRows(const std::vector<std::vector<uint8_t>>& rows) const;

 private:
  core::DecompositionTable BuildSelectedDecompositionTable() const;
  std::string BuildDecodeKey(const std::vector<int>& erased_indices, size_t block_size) const;
  std::shared_ptr<Plan> BuildEncodePlanImpl(size_t block_size);
  std::shared_ptr<Plan> BuildDecodePlanImpl(const std::vector<int>& erased_indices, size_t block_size);

  int k_;
  int r_;
  int n_;
  braid_options_t options_;
  BackendCaps caps_;
  Topology topology_;
  std::unique_ptr<ThreadPool> thread_pool_;
  core::DecompositionTable table_;
  core::Matrix generator_;
  std::unordered_map<size_t, std::shared_ptr<Plan>> encode_plan_cache_;
  std::unordered_map<std::string, std::shared_ptr<Plan>> decode_plan_cache_;
  std::unordered_map<size_t, std::shared_ptr<Plan>> encode_batch_plan_cache_;
  std::unordered_map<std::string, std::shared_ptr<Plan>> decode_batch_plan_cache_;
  std::mutex plan_cache_mutex_;
};

}  // namespace braid::runtime

#endif  // BRAID_RUNTIME_CONTEXT_H_
