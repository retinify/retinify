// Copyright (C) 2024 retinify project team. All rights reserved.
//
// This file is part of retinify.
//
// retinify is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// retinify is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with retinify.
// If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
namespace retinify
{
class Mutex
{
  public:
    inline void Lock()
    {
        std::lock_guard<std::mutex> lock(mtx_);
    }

    template <typename Predicate> inline void Wait(int ms, Predicate pred)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait_for(lock, std::chrono::milliseconds(ms), pred);
    }

    inline void Notify()
    {
        cond_.notify_all();
    }

  private:
    std::mutex mtx_;
    std::condition_variable cond_;
};
} // namespace retinify