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
#include <condition_variable>
#include <mutex>
#include <queue>
namespace retinify
{
template <typename T> class Queue
{
  public:
    Queue() = default;
    ~Queue() = default;

    /// @brief Push data to the queue
    /// @param value
    inline void Push(std::unique_ptr<T> value)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(value));
        }
        cond_.notify_one();
    }

    /// @brief Replace the front of the queue with new data
    /// @param value
    inline void Replace(std::unique_ptr<T> value)
    {
        {
            std::queue<std::unique_ptr<T>> new_queue;
            new_queue.push(std::move(value));
            {
                std::lock_guard<std::mutex> lock(mutex_);
                std::swap(queue_, new_queue);
            }
        }
        cond_.notify_one();
    }

    /// @brief Wait and Pop data from the queue
    /// @return std::unique_ptr<T>
    inline std::unique_ptr<T> Pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        std::unique_ptr<T> tmp = std::move(queue_.front());
        queue_.pop();
        return tmp;
    }

    /// @brief Wait and Pop data from the queue with timeout
    /// @param timeout time to wait for data
    /// @return std::unique_ptr<T>
    inline std::unique_ptr<T> Pop(const std::chrono::milliseconds& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return nullptr;
        }
        std::unique_ptr<T> tmp = std::move(queue_.front());
        queue_.pop();
        return tmp;
    }

    /// @brief Try to Pop data from the queue
    /// @return std::unique_ptr<T>
    inline std::unique_ptr<T> TryPop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty())
        {
            return nullptr;
        }
        std::unique_ptr<T> tmp = std::move(queue_.front());
        queue_.pop();
        return tmp;
    }

  private:
    std::queue<std::unique_ptr<T>> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
};
} // namespace retinify