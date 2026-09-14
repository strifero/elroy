"""Writes sft/handwritten.jsonl: the identity, capability and beginner-concept
examples that go into the fine-tuning set alongside the public datasets.

    python sft/make_handwritten.py

Edit this file, not the jsonl. Keep answers short, correct, and in the voice
you want the model to have; every one of these is repeated several times in
training, so a mistake here is a mistake the model learns well.
"""

import json
import os

EXAMPLES: list[tuple[str, str]] = []


def add(user: str, assistant: str) -> None:
    EXAMPLES.append((user.strip(), assistant.strip()))


# ---------------------------------------------------------------- identity
add("Who are you?", "I am Elroy, a small Python assistant. I was built from scratch as a teaching project by Strife Technologies: a 360 million parameter transformer trained on educational Python and English. I can write short Python functions, explain beginner concepts, and complete code inside a file. I am not a general assistant and I make mistakes, so run what I write.")
add("What is your name?", "Elroy. I am a small language model for Python, built from scratch by Strife Technologies.")
add("Are you ChatGPT?", "No. I am Elroy, a small open model trained from scratch by Strife Technologies as a teaching project. I am much smaller than ChatGPT and I only really know Python.")
add("What can you do?", "I can write short Python functions from a description, explain what a piece of Python does, explain beginner concepts like loops, lists, dictionaries and functions, and fill in the middle of a file when given the code before and after. I cannot browse the web, run code, or remember previous conversations, and I am unreliable on anything that is not Python.")
add("What are your limitations?", "I am a 360 million parameter model trained on about 20 billion tokens, which is tiny by modern standards. I make mistakes in code that looks plausible, I do not know recent libraries well, I cannot run code to check my answers, and I do not know much outside Python. Treat what I write as a draft and test it.")
add("Who made you?", "Strife Technologies built me from scratch as a documented teaching project, so that anyone can see every step of training a language model. The code, the data sources and the training logs are public.")
add("How were you trained?", "I was pretrained on about 20 billion tokens, roughly two thirds Python source code and one third English text, on two workstation GPUs for about a week. Then I was fine-tuned on a few hundred thousand programming questions and answers so that I reply in this format instead of just continuing text.")
add("Can you write JavaScript?", "Not well. I was trained almost entirely on Python. I will try if you ask, but expect mistakes. For Python I am much more reliable.")
add("Hi", "Hi. I am Elroy, a small Python assistant. Ask me for a function or an explanation and I will do my best.")
add("Hello!", "Hello. What Python problem can I help with?")
add("Thanks!", "You are welcome. Test the code before you rely on it.")

# ---------------------------------------------------------------- concepts
add("What is a list comprehension?", '''A list comprehension builds a list in one expression instead of a loop with `append`.

```python
squares = [x * x for x in range(5)]      # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]
```

The general shape is `[expression for item in iterable if condition]`. The `if` part is optional. It reads left to right the way you would describe it: "x squared, for each x in range(5)".''')
add("What is the difference between a list and a tuple?", '''Both hold an ordered sequence of items. The difference is that a list can be changed after it is created and a tuple cannot.

```python
a = [1, 2, 3]
a.append(4)        # fine
b = (1, 2, 3)
b.append(4)        # AttributeError: tuples have no append
```

Use a list when the contents will grow or change. Use a tuple for a fixed record, like a coordinate `(x, y)`, or when you need something hashable to use as a dictionary key.''')
add("Explain what a dictionary is in Python.", '''A dictionary maps keys to values. You look things up by key instead of by position.

```python
ages = {"ann": 31, "bob": 27}
ages["cy"] = 40            # add or update
print(ages["ann"])         # 31
print(ages.get("dan", 0))  # 0, because "dan" is not there
for name, age in ages.items():
    print(name, age)
```

Keys must be immutable (strings, numbers, tuples). Lookup is fast regardless of how many entries there are.''')
add('What does `if __name__ == "__main__":` do?', '''It runs the block only when the file is executed directly, not when it is imported.

When Python runs a file, it sets the variable `__name__` to `"__main__"`. When the same file is imported from somewhere else, `__name__` is the module's name instead. So:

```python
def main():
    print("running")

if __name__ == "__main__":
    main()
```

`python script.py` prints "running". `import script` from another file defines `main` but does not call it.''')
add("What is the difference between `==` and `is`?", '''`==` asks whether two values are equal. `is` asks whether two names refer to the same object in memory.

```python
a = [1, 2]
b = [1, 2]
a == b   # True, same contents
a is b   # False, two separate lists
```

Use `==` almost always. Use `is` only for singletons like `None`: `if x is None:`.''')
add("How do I read a file line by line?", '''Open it in a `with` block and iterate over the file object:

```python
with open("data.txt") as f:
    for line in f:
        line = line.rstrip("\\n")
        print(line)
```

The `with` block closes the file when you leave it, even if an error occurs. Iterating the file object reads one line at a time, so this works for files larger than memory.''')
add("What is a generator?", '''A generator is a function that produces values one at a time with `yield` instead of building a whole list and returning it.

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for x in countdown(3):
    print(x)        # 3, 2, 1
```

Each time the loop asks for the next value, the function resumes where it left off. Generators use almost no memory no matter how many values they produce, which is why `range`, file iteration and most of `itertools` work this way.''')
add("What is the difference between `append` and `extend`?", '''`append` adds one item. `extend` adds every item from another iterable.

```python
a = [1, 2]
a.append([3, 4])   # [1, 2, [3, 4]]  one item, which happens to be a list
b = [1, 2]
b.extend([3, 4])   # [1, 2, 3, 4]
```''')
add("Explain try/except.", '''`try`/`except` lets you handle an error instead of crashing.

```python
try:
    n = int(input("number: "))
except ValueError:
    print("that was not a number")
    n = 0
```

Python runs the `try` block. If a `ValueError` is raised inside it, execution jumps to the `except` block. Catch specific exceptions rather than a bare `except:`, so that real bugs still surface. `finally:` runs either way, and `else:` runs only if nothing was raised.''')
add("What does `*args` and `**kwargs` mean?", '''They let a function accept any number of arguments.

```python
def f(*args, **kwargs):
    print(args)     # a tuple of the positional arguments
    print(kwargs)   # a dict of the keyword arguments

f(1, 2, x=3)        # (1, 2) then {'x': 3}
```

`*args` collects extra positional arguments into a tuple; `**kwargs` collects extra keyword arguments into a dictionary. The names `args` and `kwargs` are convention; the stars are what matter. The same stars unpack in the other direction when calling: `f(*[1, 2], **{"x": 3})`.''')
add("What is a lambda?", '''A lambda is a small anonymous function written inline.

```python
square = lambda x: x * x
sorted(words, key=lambda w: len(w))
```

It can only contain a single expression, and its value is returned. Use it for short throwaway functions passed to `sorted`, `map` or `filter`. Anything longer than one line should be a normal `def`.''')
add("How do I reverse a string in Python?", '''Slice it with a step of -1:

```python
s = "hello"
print(s[::-1])   # olleh
```

`[::-1]` means start to end, stepping backwards one character at a time. Strings are immutable, so this returns a new string.''')
add("What is recursion?", '''Recursion is when a function calls itself on a smaller version of the problem, until it reaches a case small enough to answer directly.

```python
def factorial(n):
    if n <= 1:          # base case
        return 1
    return n * factorial(n - 1)   # recursive case
```

Every recursive function needs a base case that stops the recursion, otherwise it runs until Python raises `RecursionError`. Anything recursive can also be written as a loop; recursion is clearest for things that are naturally nested, like trees.''')
add("Explain what `enumerate` does.", '''`enumerate` gives you the index alongside each item when looping:

```python
for i, name in enumerate(["ann", "bob"]):
    print(i, name)
# 0 ann
# 1 bob
```

Pass `start=1` to count from 1. It replaces the pattern `for i in range(len(items)): item = items[i]`, which is longer and easier to get wrong.''')
add("What is the difference between a function and a method?", '''A method is a function that belongs to an object. You call it on the object with a dot, and the object is passed in automatically as the first argument, `self`.

```python
def shout(text):          # function
    return text.upper()

class Dog:
    def speak(self):      # method
        return "woof"

shout("hi")               # 'HI'
Dog().speak()             # 'woof'
```

`"hi".upper()` is a method call on the string; `len("hi")` is a plain function call.''')
add("How do I write a function with a default argument?", '''Give the parameter a value in the signature:

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Ann")               # 'Hello, Ann!'
greet("Ann", "Hi")         # 'Hi, Ann!'
```

Parameters with defaults must come after those without. One trap: never use a mutable default like `def f(items=[])`, because the same list is shared between calls. Use `items=None` and create the list inside the function.''')
add("What is PEP 8?", "PEP 8 is the official style guide for Python code. The main rules: indent with four spaces, keep lines under 79 characters (many projects use 88 or 100), name functions and variables in `snake_case` and classes in `CamelCase`, put two blank lines between top-level definitions, and put spaces around operators and after commas. Tools like `ruff`, `flake8` and `black` check or apply it automatically. Following it matters less than being consistent within a project.")
add("Is Python pass by value or pass by reference?", '''Neither, exactly. Python passes the object reference by value: the function gets a new name bound to the same object.

```python
def add_item(items):
    items.append(1)      # mutates the shared list; the caller sees it

def rebind(items):
    items = [1]          # rebinds the local name only; the caller does not see it
```

So mutating an argument is visible outside the function, but reassigning the parameter is not. Immutable objects like ints and strings cannot be mutated, which is why they behave like pass by value.''')
add("Why do I get 'IndexError: list index out of range'?", '''You asked for a position that does not exist in the list. Indexes start at 0, so a list of 3 items has positions 0, 1 and 2; asking for `items[3]` raises this error.

Common causes:

```python
items = [10, 20, 30]
items[3]              # off by one: the last valid index is len(items) - 1
items[len(items)]     # same mistake
for i in range(1, len(items) + 1):   # loop should be range(len(items))
    print(items[i])
```

Print `len(items)` and the index right before the failing line to see which it is. If the list can be empty, check `if items:` before indexing.''')
add("How do I sort a list of dictionaries by one key?", '''Pass a `key` function to `sorted` (or `list.sort`):

```python
people = [{"name": "ann", "age": 31}, {"name": "bob", "age": 27}]
by_age = sorted(people, key=lambda p: p["age"])
```

`sorted` returns a new list; `people.sort(key=...)` sorts in place. Add `reverse=True` for descending order. `operator.itemgetter("age")` does the same job as the lambda and is slightly faster.''')
add("How do I convert a string to an integer safely?", '''Wrap `int()` in `try/except`, because it raises `ValueError` on anything that is not a whole number:

```python
def to_int(s, default=None):
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return default


to_int("42")     # 42
to_int(" 7 ")    # 7
to_int("4.5")    # None
```

`AttributeError` covers the case where `s` is not a string at all. `int()` does not accept decimals like "4.5"; use `float()` first if you want to allow them.''')
add("What does this code do?\n\n```python\nresult = {k: v for k, v in pairs if v > 0}\n```", '''It builds a dictionary from `pairs`, keeping only the entries whose value is positive.

`pairs` is expected to be an iterable of `(key, value)` tuples, for example `[("a", 1), ("b", -2)]`. The dictionary comprehension iterates over them, unpacks each into `k` and `v`, and includes `k: v` in the result only when `v > 0`. For that example the result is `{"a": 1}`.''')

# ---------------------------------------------------------- small functions
add("Write a function that checks whether a number is prime.", '''```python
def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
```

It rejects numbers below 2, handles 2 as the only even prime, then tests odd divisors up to the square root of `n`. Any factor larger than the square root has a partner smaller than it, so there is no need to look further.''')
add("Write fizzbuzz.", '''```python
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

fizzbuzz(15)
```

The `% 15` check has to come first: a number divisible by both 3 and 5 would otherwise match the `% 3` branch and print only "Fizz".''')
add("Write a function to reverse a linked list.", '''```python
class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


def reverse(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next     # remember where we were going
        cur.next = prev    # point this node backwards
        prev = cur         # advance
        cur = nxt
    return prev
```

Walk the list once, flipping each `next` pointer to face the previous node. `prev` ends up as the old tail, which is the new head. Runs in O(n) time and O(1) extra space.''')
add("Write a function that counts how many times each word appears in a string.", '''```python
from collections import Counter


def word_counts(text):
    words = text.lower().split()
    return Counter(words)


print(word_counts("the cat and the hat"))
# Counter({'the': 2, 'cat': 1, 'and': 1, 'hat': 1})
```

`split()` breaks on whitespace, `lower()` makes "The" and "the" count together, and `Counter` does the tallying. If you need to strip punctuation too, run each word through `word.strip(".,!?")` first.''')
add("Write a function that reads a CSV file and returns the sum of a column.", '''```python
import csv


def column_sum(path, column):
    total = 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total += float(row[column])
    return total


print(column_sum("sales.csv", "amount"))
```

`csv.DictReader` uses the header row as keys, so you can refer to the column by name. Values come in as strings, hence the `float()`. Rows with a blank or non-numeric value will raise `ValueError`; wrap the conversion in `try/except` if your file has those.''')
add("How do I find the largest number in a list?", '''Use the built-in `max`:

```python
numbers = [3, 9, 2, 7]
print(max(numbers))   # 9
```

If you want to see how it works underneath:

```python
def largest(numbers):
    best = numbers[0]
    for n in numbers[1:]:
        if n > best:
            best = n
    return best
```

Both raise an error on an empty list; `max(numbers, default=None)` returns `None` instead.''')
add("Write a function that returns the n-th Fibonacci number.", '''```python
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

`fib(0)` is 0, `fib(1)` is 1, `fib(10)` is 55. The tuple assignment advances both values in one step without a temporary variable. This runs in O(n) time; the naive recursive version is exponential.''')
add("Write a function to check if a string is a palindrome, ignoring case and spaces.", '''```python
def is_palindrome(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


print(is_palindrome("A man, a plan, a canal: Panama"))   # True
```

Keep only letters and digits, lowercase them, then compare the string with its reverse.''')
add("Write a function that merges two sorted lists into one sorted list.", '''```python
def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


print(merge([1, 4, 9], [2, 3, 10]))   # [1, 2, 3, 4, 9, 10]
```

Walk both lists with a pointer each, always taking the smaller head. When one list runs out, append whatever is left of the other. This is the merge step of merge sort and runs in O(len(a) + len(b)).''')
add("Write a function that removes duplicates from a list while keeping the original order.", '''```python
def dedupe(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


print(dedupe([3, 1, 3, 2, 1]))   # [3, 1, 2]
```

A set gives O(1) membership checks, and the list preserves the order of first appearance. `list(dict.fromkeys(items))` is a one-line version that relies on dictionaries keeping insertion order.''')
add("Write a class representing a bank account with deposit and withdraw methods.", '''```python
class BankAccount:
    def __init__(self, owner, balance=0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("withdrawal must be positive")
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount

    def __repr__(self):
        return f"BankAccount({self.owner!r}, {self.balance:.2f})"
```

The checks raise `ValueError` rather than silently ignoring bad input, which makes bugs visible at the call site. `__repr__` gives a readable form when you print the object.''')
add("Complete this function:\n\n```python\ndef average(numbers):\n```", '''```python
def average(numbers):
    if not numbers:
        raise ValueError("average of an empty list is undefined")
    return sum(numbers) / len(numbers)
```

The empty-list check avoids a `ZeroDivisionError` with a clearer message. If you would rather get `0` for an empty list, return that instead of raising.''')


def main() -> None:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handwritten.jsonl")
    with open(out, "w") as f:
        for user, assistant in EXAMPLES:
            f.write(json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n")
    print(f"wrote {len(EXAMPLES)} examples to {out}")


if __name__ == "__main__":
    main()
