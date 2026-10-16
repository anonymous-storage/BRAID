#include "src/runtime/thread_pool.h"
#include <atomic>
#include <stdexcept>
int main() {
  braid::runtime::ThreadPool pool({-1, -1, -1, -1});
  for (bool spin : {false, true}) {
    for (int culprit : {0, 1, 3}) {
      bool caught = false;
      try { pool.Run(4, [=](int id) { if (id == culprit) throw std::runtime_error("injected"); }, spin); }
      catch (const std::runtime_error&) { caught = true; }
      if (!caught) return 1;
      std::atomic<int> completed{0};
      pool.Run(4, [&](int) { ++completed; }, spin);
      if (completed != 4) return 2;
    }
  }
}
