# What a small model can and cannot do

*Draft 4 for strifetech.com. Week four. Links to Chapters 5 and 6. Needs the eval numbers and three or four real samples from the trained model.*

Elroy, the model we trained from scratch, is small. 360 million parameters, against the hundreds of billions in the models you use every day. We built it to understand the technology, not to compete with those models, and the most useful thing we can do with it now is describe exactly where the edge of its ability is. Vendors rarely do this. We think it is the most important part of the write-up.

## What it does well

Ask Elroy for a function that checks whether a number is prime, reverses a linked list, counts words in a string, or reads a column from a CSV file, and it writes one that works. Ask it to explain a list comprehension, the difference between a list and a tuple, or why a particular error message appears, and it gives a short, correct explanation with an example. Give it the code before and after a gap in a file and it fills the gap.

On the two standard coding tests, HumanEval and MBPP, it scores [X]% and [Y]%. Those tests present a function description and check whether the model's code passes hidden tests. A score of [X]% means [X] of every 100 such tasks came back fully correct on the first try, with no retries and no hints.

## What it does badly

Anything that requires holding more than a few steps of logic at once. Anything outside Python. Anything that depends on a library released after its training data was collected. Anything factual about the world, where it will produce confident, fluent, wrong answers, because it was trained mostly on code and educational text and has no mechanism for knowing what it does not know. Long conversations, because it has no memory between questions. And it has not been trained to refuse anything, so it will attempt whatever it is asked and fail quietly rather than decline.

Here is a representative failure, unedited:

> [sample of a plausible-looking wrong answer]

It looks right. It is not. Nothing in the output signals the difference. This is the property of these systems that matters most in a professional setting, and it does not go away in the large models; it gets harder to spot.

## Why we made it narrow on purpose

Elroy is two thirds Python by training and its vocabulary was built for code. That decision made it worse at English and better at its job than a general model of the same size would be. It is the same trade we would recommend to most firms considering their own AI tooling: a small model that does one thing under your control, with a documented failure profile, is more useful and more governable than a large one that does everything and explains nothing.

## How we measured it

Every number above comes from a test harness in the repository that anyone can run. It generates the model's answer, runs the answer against the test cases in an isolated process with time and memory limits, and reports the pass rate. We report the base model and the chat model separately and we report the numbers that disappointed us. A benchmark you cannot reproduce is a claim, not a measurement.

The samples, the curves and the full evaluation are in Chapters 5 and 6.
