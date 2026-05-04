# Five Lines of Code Rules (ALWAYS OBEY)

1. **Five-Line Limit:** No function body exceeds 5 lines (excluding signature/braces). Decompose if needed.
2. **No `else`:** Use guard clauses, early returns, or Strategy pattern.
3. **No `switch`:** Use polymorphism, map objects, or specialized classes.
4. **One Statement Per Line:** No semicolon packing, no long method chains.
5. **No Complex Ternaries:** Avoid nested/obscure ternary operators.
6. **If Only at Start:** `if` statements only for edge-case guards at function start.
7. **Single Abstraction Level:** Functions either do work OR coordinate, never both.
8. **Either Call or Pass:** On an object, either call methods OR pass it, not both.
9. **Do not use magic constants.** Always name a constant to a meaningful name.