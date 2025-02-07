## Optimizations

- Remove member `pairs` in `NgramFinder`. Use local variable `pair_map` instead.
- Use `tuple` instead of `vector` for `pair_map`. Add a corresponding hashmap.
- Bugfix: use reference instead of value in lambda function.