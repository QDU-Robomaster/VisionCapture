#pragma once

#include <cerrno>
#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#if !defined(_WIN32)
#include <poll.h>
#include <unistd.h>
#endif

namespace VisionCaptureDetail
{
/**
 * @brief A joinable line reader whose stop path never waits for a newline or EOF.
 *
 * The file descriptor is borrowed and remains owned by the caller. Complete lines are
 * delivered without the trailing '\n'; a final partial line is delivered only on EOF.
 */
class StoppableLineInput
{
 public:
  using Handler = std::function<void(std::string_view)>;
  using FailureHandler = std::function<void(std::exception_ptr)>;

  StoppableLineInput() = default;
  StoppableLineInput(const StoppableLineInput&) = delete;
  StoppableLineInput& operator=(const StoppableLineInput&) = delete;

  ~StoppableLineInput() { Stop(); }

  [[nodiscard]] bool StartStandardInput(Handler handler,
                                        FailureHandler failure_handler = {})
  {
#if defined(_WIN32)
    (void)handler;
    (void)failure_handler;
    return false;
#else
    return Start(STDIN_FILENO, std::move(handler), std::move(failure_handler));
#endif
  }

  [[nodiscard]] bool Start(int fd, Handler handler, FailureHandler failure_handler = {})
  {
#if defined(_WIN32)
    (void)fd;
    (void)handler;
    (void)failure_handler;
    return false;
#else
    if (fd < 0 || !handler || worker_.joinable())
    {
      return false;
    }

    try
    {
      worker_ = std::jthread(
          [fd, handler = std::move(handler),
           failure_handler = std::move(failure_handler)](std::stop_token stop) mutable
          {
            try
            {
              ReadLoop(fd, stop, handler);
            }
            catch (...)
            {
              if (failure_handler)
              {
                try
                {
                  failure_handler(std::current_exception());
                }
                catch (...)
                {
                }
              }
            }
          });
    }
    catch (...)
    {
      return false;
    }
    return true;
#endif
  }

  void Stop()
  {
    if (!worker_.joinable())
    {
      return;
    }
    worker_.request_stop();
    worker_.join();
  }

 private:
#if !defined(_WIN32)
  static constexpr int kPollTimeoutMs = 50;
  static constexpr std::size_t kReadBufferBytes = 256;

  static void DispatchCompleteLines(std::string& pending, std::stop_token stop,
                                    Handler& handler)
  {
    std::size_t consumed = 0;
    while (!stop.stop_requested())
    {
      const std::size_t newline = pending.find('\n', consumed);
      if (newline == std::string::npos)
      {
        break;
      }
      handler(std::string_view(pending).substr(consumed, newline - consumed));
      consumed = newline + 1U;
    }
    if (consumed != 0U)
    {
      pending.erase(0, consumed);
    }
  }

  static void ReadLoop(int fd, std::stop_token stop, Handler& handler)
  {
    std::string pending;
    char buffer[kReadBufferBytes];
    while (!stop.stop_requested())
    {
      pollfd descriptor{fd, POLLIN, 0};
      int ready = 0;
      do
      {
        ready = ::poll(&descriptor, 1, kPollTimeoutMs);
      } while (ready < 0 && errno == EINTR && !stop.stop_requested());

      if (stop.stop_requested())
      {
        return;
      }
      if (ready < 0)
      {
        throw std::system_error(errno, std::generic_category(), "poll stdin");
      }
      if (ready == 0)
      {
        continue;
      }
      if ((descriptor.revents & POLLNVAL) != 0)
      {
        throw std::system_error(EBADF, std::generic_category(), "poll stdin");
      }
      if ((descriptor.revents & POLLERR) != 0)
      {
        throw std::system_error(EIO, std::generic_category(), "poll stdin");
      }
      if ((descriptor.revents & (POLLIN | POLLHUP)) == 0)
      {
        continue;
      }

      ssize_t count = 0;
      do
      {
        count = ::read(fd, buffer, sizeof(buffer));
      } while (count < 0 && errno == EINTR && !stop.stop_requested());

      if (stop.stop_requested())
      {
        return;
      }
      if (count == 0)
      {
        if (!stop.stop_requested() && !pending.empty())
        {
          handler(pending);
        }
        return;
      }
      if (count < 0)
      {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
          continue;
        }
        throw std::system_error(errno, std::generic_category(), "read stdin");
      }

      pending.append(buffer, static_cast<std::size_t>(count));
      DispatchCompleteLines(pending, stop, handler);
    }
  }
#endif

  std::jthread worker_{};
};
}  // namespace VisionCaptureDetail
