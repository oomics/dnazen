## Optimize Memory Usage of NgramFinder

- using `uint32_t` instead of `size_t` to count frequencies
- free the space of `std:unordered_map` used to count pair frequency after it is no longer used.