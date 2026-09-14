# What is actually in the training data

*Draft 2 for strifetech.com. Week two. Links to Chapter 1 of the Elroy write-up.*

When we set out to train Elroy, our own small language model, we expected the hard part to be the model. It was not. The model is 250 lines of code that follow a published recipe. The hard part, and the part with all the risk in it, was the data.

A language model is a compression of its training data. It cannot know anything that was not in there, and it will faithfully reproduce whatever was, including the mistakes. So before we trained anything we did what surprisingly few people do: we opened the data and read it.

## Four sources, all public

Elroy learned from about 120 gigabytes of text drawn from four public datasets: a collection of educational Python code filtered by a quality classifier, a second collection of permissively licensed Python from GitHub, a set of educational web pages, and a set of synthetic textbook-style articles. Each has a license that allows what we did with it, and each is named on the model's card with that license.

One well-known source was left out on purpose. Stack Overflow's question-and-answer data would have been ideal for teaching the model to answer programming questions, but it is published under a share-alike license. Training on it would arguably attach that license to the model itself. We chose not to find out.

## The first surprise: the corpus was an index

The largest dataset, 25 million Python files, turned out not to contain any code. Each row was a pointer, an identifier for a file in a public software archive, plus a quality score. The actual text had to be fetched, one file at a time, from a public storage bucket. Getting 40 gigabytes of it took an afternoon and a lesson in network engineering that is in the technical write-up.

The point for a non-engineer is this: "we trained on dataset X" can mean the vendor downloaded X and used it, or it can mean they resolved millions of references, some of which no longer exist, made choices about which ones to keep, and applied filters nobody wrote down. The same name covers both.

## The second surprise: half the files were broken

We wrote a small check that asks a simple question of every code file: does Python itself accept this as valid code? For the GitHub dataset, the answer was yes for only 48% of files.

That number should have been over 90%. The cause was mundane. The dataset had been prepared for a specific model, and that preparation had left tags at the top of nearly half the files: a repository name, a filename, a star count, each wrapped in angle brackets. Not Python, not documentation, just leftover metadata. Left in, our model would have learned that Python programs begin with a line of garbage, and it would have written that line every time.

Stripping the tags with one rule took the pass rate to 94%. The remaining 6% was mostly code written for Python 2, which has been unsupported since 2020, and we dropped it too. A model trained only on valid modern Python is more likely to write valid modern Python.

Nobody had hidden any of this. It was all there in the first row of the first file. It only required looking.

## Why this matters to you

Every AI product your firm evaluates was trained on data that went through choices like these, made by people you will never meet, mostly undocumented. The questions worth asking a vendor are not about accuracy scores. They are: what was it trained on, under what license, who checked it, and what did they throw away? A vendor who can answer those in detail has done the work. A vendor who answers with "high-quality, curated data" has given you a phrase.

For Elroy the answers are in Chapter 1 of the write-up, including the exact filter rules and the counts of what each one removed.
