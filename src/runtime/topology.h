#ifndef BRAID_RUNTIME_TOPOLOGY_H_
#define BRAID_RUNTIME_TOPOLOGY_H_

#include <vector>

namespace braid::runtime {

struct Topology {
  bool valid = false;
  std::vector<int> primary_cpus;
  std::vector<int> all_cpus;
  std::vector<std::vector<int>> primary_cpus_by_node;
  std::vector<std::vector<int>> all_cpus_by_node;
};

Topology DetectTopology();
std::vector<int> BuildCpuBindingOrder(
    const Topology& topology, int requested_threads, bool interleave_nodes = true);
bool PinCurrentThreadToCpu(int cpu);

class AffinityGuard {
 public:
  AffinityGuard();
  ~AffinityGuard();

  bool PinToCpu(int cpu);

 private:
  bool captured_ = false;
  bool changed_ = false;
  unsigned char storage_[256];
};

}  // namespace braid::runtime

#endif  // BRAID_RUNTIME_TOPOLOGY_H_
