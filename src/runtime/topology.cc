#include "src/runtime/topology.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <pthread.h>
#include <sched.h>

namespace braid::runtime {

namespace {

struct CpuEntry {
  int cpu = -1;
  int node = -1;
  int core = -1;
};

bool ReadInt(const std::filesystem::path& path, int* value) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return false;
  }
  input >> *value;
  return !input.fail();
}

int DetectNodeForCpu(const std::filesystem::path& cpu_path) {
  for (const auto& entry : std::filesystem::directory_iterator(cpu_path)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("node", 0) == 0 && name.size() > 4) {
      try {
        return std::stoi(name.substr(4));
      } catch (...) {
        return 0;
      }
    }
  }
  return 0;
}

std::vector<int> Interleave(const std::vector<std::vector<int>>& by_node) {
  std::vector<int> order;
  size_t max_size = 0;
  for (const auto& cpus : by_node) {
    max_size = std::max(max_size, cpus.size());
  }
  for (size_t index = 0; index < max_size; ++index) {
    for (const auto& cpus : by_node) {
      if (index < cpus.size()) {
        order.push_back(cpus[index]);
      }
    }
  }
  return order;
}

}  // namespace

Topology DetectTopology() {
  Topology topology;
  std::vector<CpuEntry> entries;
  cpu_set_t allowed_cpus;
  CPU_ZERO(&allowed_cpus);
  const bool have_affinity = sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) == 0;
  const std::filesystem::path cpu_root("/sys/devices/system/cpu");
  for (const auto& entry : std::filesystem::directory_iterator(cpu_root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.rfind("cpu", 0) != 0 || name.size() <= 3) {
      continue;
    }
    bool digits_only = true;
    for (size_t i = 3; i < name.size(); ++i) {
      if (name[i] < '0' || name[i] > '9') {
        digits_only = false;
        break;
      }
    }
    if (!digits_only) {
      continue;
    }

    CpuEntry cpu;
    cpu.cpu = std::stoi(name.substr(3));
    if (have_affinity && (cpu.cpu >= CPU_SETSIZE || !CPU_ISSET(cpu.cpu, &allowed_cpus))) {
      continue;
    }
    cpu.node = DetectNodeForCpu(entry.path());
    if (!ReadInt(entry.path() / "topology/core_id", &cpu.core)) {
      cpu.core = cpu.cpu;
    }
    entries.push_back(cpu);
  }

  if (entries.empty()) {
    return topology;
  }

  std::sort(entries.begin(), entries.end(), [](const CpuEntry& lhs, const CpuEntry& rhs) {
    return lhs.cpu < rhs.cpu;
  });

  std::map<int, std::map<int, std::vector<int>>> by_node_core;
  for (const auto& entry : entries) {
    by_node_core[entry.node][entry.core].push_back(entry.cpu);
  }

  std::vector<std::vector<int>> primary_by_node;
  std::vector<std::vector<int>> siblings_by_node;
  for (auto& [node, cores] : by_node_core) {
    std::vector<int> primaries;
    std::vector<int> siblings;
    for (auto& [core, cpus] : cores) {
      std::sort(cpus.begin(), cpus.end());
      if (!cpus.empty()) {
        primaries.push_back(cpus.front());
        for (size_t i = 1; i < cpus.size(); ++i) {
          siblings.push_back(cpus[i]);
        }
      }
    }
    std::sort(primaries.begin(), primaries.end());
    std::sort(siblings.begin(), siblings.end());
    primary_by_node.push_back(std::move(primaries));
    siblings_by_node.push_back(std::move(siblings));
  }

  topology.primary_cpus_by_node = primary_by_node;
  topology.all_cpus_by_node = primary_by_node;
  for (size_t node = 0; node < topology.all_cpus_by_node.size(); ++node) {
    topology.all_cpus_by_node[node].insert(
        topology.all_cpus_by_node[node].end(),
        siblings_by_node[node].begin(),
        siblings_by_node[node].end());
  }
  topology.primary_cpus = Interleave(primary_by_node);
  topology.all_cpus = topology.primary_cpus;
  const auto sibling_order = Interleave(siblings_by_node);
  topology.all_cpus.insert(topology.all_cpus.end(), sibling_order.begin(), sibling_order.end());
  topology.valid = !topology.primary_cpus.empty();
  return topology;
}

std::vector<int> BuildCpuBindingOrder(
    const Topology& topology, int requested_threads, bool interleave_nodes) {
  if (!topology.valid || requested_threads <= 0) {
    return {};
  }
  const bool use_primary = requested_threads <= static_cast<int>(topology.primary_cpus.size());
  std::vector<int> packed;
  const std::vector<int>* source = nullptr;
  if (interleave_nodes) {
    source = use_primary ? &topology.primary_cpus : &topology.all_cpus;
  } else {
    const auto& by_node = use_primary ? topology.primary_cpus_by_node : topology.all_cpus_by_node;
    for (const auto& node_cpus : by_node) {
      packed.insert(packed.end(), node_cpus.begin(), node_cpus.end());
    }
    source = &packed;
  }
  std::vector<int> order;
  order.reserve(std::min<int>(requested_threads, static_cast<int>(source->size())));
  for (int i = 0; i < requested_threads && i < static_cast<int>(source->size()); ++i) {
    order.push_back((*source)[static_cast<size_t>(i)]);
  }
  return order;
}

bool PinCurrentThreadToCpu(int cpu) {
  if (cpu < 0 || cpu >= CPU_SETSIZE) {
    return false;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

AffinityGuard::AffinityGuard() {
  auto* set = reinterpret_cast<cpu_set_t*>(storage_);
  captured_ = pthread_getaffinity_np(pthread_self(), sizeof(storage_), set) == 0;
}

AffinityGuard::~AffinityGuard() {
  if (captured_ && changed_) {
    auto* set = reinterpret_cast<const cpu_set_t*>(storage_);
    pthread_setaffinity_np(pthread_self(), sizeof(storage_), set);
  }
}

bool AffinityGuard::PinToCpu(int cpu) {
  changed_ = PinCurrentThreadToCpu(cpu);
  return changed_;
}

}  // namespace braid::runtime
