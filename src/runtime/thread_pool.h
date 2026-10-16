#ifndef BRAID_RUNTIME_THREAD_POOL_H_
#define BRAID_RUNTIME_THREAD_POOL_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "src/runtime/topology.h"

namespace braid::runtime {

inline void ThreadPoolPause() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ volatile("yield");
#endif
}

class ThreadPool {
 public:
  explicit ThreadPool(std::vector<int> cpu_order);
  ~ThreadPool();

  int capacity() const { return std::max<int>(1, static_cast<int>(cpu_order_.size())); }

  template <typename Fn>
  void Run(int threads, Fn&& fn, bool spin_wait = false) {
    std::lock_guard<std::mutex> run_lock(run_mutex_);
    if (threads <= 1 || workers_.empty()) {
      AffinityGuard guard;
      if (!cpu_order_.empty()) {
        guard.PinToCpu(cpu_order_.front());
      }
      fn(0);
      return;
    }

    const int run_threads = std::min<int>(threads, static_cast<int>(cpu_order_.size()));
    if (run_threads <= 1) {
      fn(0);
      return;
    }

    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_ = std::forward<Fn>(fn);
      run_threads_ = run_threads;
      remaining_workers_ = run_threads - 1;
      error_ = nullptr;
      completed_workers_.store(0, std::memory_order_relaxed);
      spin_workers_ = spin_wait;
      ++generation_;
      published_generation_.store(generation_, std::memory_order_release);
    }
    cv_start_.notify_all();

    AffinityGuard guard;
    if (!cpu_order_.empty()) {
      guard.PinToCpu(cpu_order_.front());
    }
    try { fn(0); } catch (...) { RecordException(); }

    if (spin_wait) {
      for (int spin = 0; spin < 8192; ++spin) {
        if (completed_workers_.load(std::memory_order_acquire) >= run_threads - 1) {
          break;
        }
        ThreadPoolPause();
      }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [&] { return remaining_workers_ == 0; });
    task_ = nullptr;
    if (error_) std::rethrow_exception(error_);
  }

 private:
  void WorkerLoop(int worker_id);
  void RecordException() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_) error_ = std::current_exception();
  }

  std::vector<int> cpu_order_;
  std::vector<std::thread> workers_;
  std::mutex run_mutex_;
  std::mutex mutex_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  std::function<void(int)> task_;
  std::exception_ptr error_;
  bool stop_ = false;
  size_t generation_ = 0;
  std::atomic<size_t> published_generation_{0};
  int run_threads_ = 1;
  int remaining_workers_ = 0;
  std::atomic<int> completed_workers_{0};
  bool spin_workers_ = false;
};

}  // namespace braid::runtime

#endif  // BRAID_RUNTIME_THREAD_POOL_H_
