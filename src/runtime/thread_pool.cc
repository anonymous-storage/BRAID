#include "src/runtime/thread_pool.h"

namespace braid::runtime {

ThreadPool::ThreadPool(std::vector<int> cpu_order) : cpu_order_(std::move(cpu_order)) {
  const int max_threads = static_cast<int>(cpu_order_.size());
  workers_.reserve(max_threads > 0 ? static_cast<size_t>(max_threads - 1) : 0);
  try {
    for (int worker_id = 1; worker_id < max_threads; ++worker_id) {
      workers_.emplace_back([this, worker_id]() { WorkerLoop(worker_id); });
    }
  } catch (...) {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    cv_start_.notify_all();
    for (auto& worker : workers_) worker.join();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
    ++generation_;
    published_generation_.store(generation_, std::memory_order_release);
  }
  cv_start_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::WorkerLoop(int worker_id) {
  if (worker_id < static_cast<int>(cpu_order_.size())) {
    PinCurrentThreadToCpu(cpu_order_[static_cast<size_t>(worker_id)]);
  }

  size_t seen_generation = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    cv_start_.wait(lock, [&] { return stop_ || generation_ != seen_generation; });
    if (stop_) {
      return;
    }

    seen_generation = generation_;
    const int run_threads = run_threads_;
    const bool spin_workers = spin_workers_;
    // Run() keeps task_ alive and unchanged until all workers finish.
    const bool have_task = static_cast<bool>(task_);
    lock.unlock();

    if (worker_id < run_threads && have_task) {
      try { task_(worker_id); } catch (...) { RecordException(); }
    }

    lock.lock();
    if (worker_id < run_threads && have_task) {
      completed_workers_.fetch_add(1, std::memory_order_release);
      --remaining_workers_;
      if (remaining_workers_ == 0) {
        cv_done_.notify_one();
      }
    }
    if (spin_workers) {
      lock.unlock();
      for (int spin = 0; spin < 8192; ++spin) {
        if (published_generation_.load(std::memory_order_acquire) != seen_generation) {
          break;
        }
        ThreadPoolPause();
      }
      lock.lock();
    }
  }
}

}  // namespace braid::runtime
