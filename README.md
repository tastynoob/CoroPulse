# CoroPulse

CoroPulse is a C++20 coroutine-driven tick simulator prototype.

It provides:

- deterministic tick execution;
- a fixed worker-thread scheduler;
- `Channel<T>` for one-tick data visibility;
- `Signal<T>` for same-tick coroutine wakeup and explicit backpressure;
- basic correctness tests and rate benchmarks.

Build and test:

```sh
make test
```

Run rate benchmarks:

```sh
make rate-test
```

Run only the signal/backpressure benchmark:

```sh
make signal-rate-test
```
