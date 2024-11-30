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

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <retinify/thread.hpp>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
class retinify::Thread::Impl
{
  public:
    Impl(std::function<void()> func)
    {
        this->function_ = std::move(func);
    }

    ~Impl()
    {
        Stop();
    }

    bool Start()
    {
        this->Stop(); // make sure the previous thread is stopped

        this->running_ = true;
        this->thread_ = std::thread(&Impl::Execution, this);
        return true;
    }

    bool Start(double frequency)
    {
        this->Stop(); // make sure the previous thread is stopped

        this->running_ = true;
        this->period_ = static_cast<unsigned long long>((1.0 / frequency) * 1e9);
        this->thread_ = std::thread(&Impl::RTExecution, this);
        return true;
    }

    bool Stop()
    {
        this->running_ = false;
        if (this->thread_.joinable())
        {
            this->thread_.join();
        }
        return true;
    }

  private:
    bool running_;
    unsigned long long period_;
    std::function<void()> function_;
    std::thread thread_;

    struct sched_attr
    {
        uint32_t size;
        uint32_t sched_policy;
        uint64_t sched_flags;
        int32_t sched_nice;
        uint32_t sched_priority;
        uint64_t sched_runtime;
        uint64_t sched_deadline;
        uint64_t sched_period;
    };

    bool Apply()
    {
        sched_attr attr{};
        attr.size = sizeof(sched_attr);
        attr.sched_policy = SCHED_DEADLINE;
        attr.sched_runtime = this->period_;
        attr.sched_deadline = this->period_;
        attr.sched_period = this->period_;

        if (syscall(__NR_sched_setattr, 0, &attr, 0) == 0)
        {
            std::cout << "\033[32mThis program is running with root privileges.\033[0m" << std::endl;
            return true;
        }
        else
        {
            std::cerr << "\033[31m[ERROR] " << strerror(errno) << "\033[0m" << std::endl;
            std::cerr << "\033[31mPlease run this program with root privileges\033[0m" << std::endl;
            return false;
        }
    }

    void RTExecution()
    {
        if (!this->Apply())
        {
            return;
        }

        while (this->running_)
        {
            auto time = std::chrono::steady_clock::now() + std::chrono::nanoseconds(this->period_);

            this->function_();

            while (std::chrono::steady_clock::now() < time)
            {
                // wait
            }
        }
    }

    void Execution()
    {
        while (this->running_)
        {
            this->function_();
        }
    }
};

retinify::Thread::Thread(std::function<void()> func)
{
    this->impl_ = std::make_unique<Impl>(func);
}

retinify::Thread::~Thread() = default;

bool retinify::Thread::Start()
{
    return this->impl_->Start();
}

bool retinify::Thread::Start(double frequency)
{
    return this->impl_->Start(frequency);
}

bool retinify::Thread::Stop()
{
    return this->impl_->Stop();
}