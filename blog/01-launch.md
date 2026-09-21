# We built a language model from scratch. Here is what it taught us.

*Draft 1 for strifetech.com. Byline: Sean Trifero, Strife Technologies. Publish with the repo and the weights on the same day. Bracketed items are filled in when the run finishes.*

Over the last few weeks we built a language model from nothing. Not a fine-tune of someone else's model, and not a wrapper around an API. Our own tokenizer, our own transformer, our own training loop, trained on two workstation graphics cards in our own rack, for about six days. We named it Elroy. It writes short Python programs, explains beginner programming concepts, and it will tell you plainly that it is a small model and that you should test what it writes.

Elroy is not going to replace anything you use today. It is about the size of the model that made headlines in 2019 as too dangerous to release. What is different is that the recipe for building one is now public, the training data is now public, and the compute that took a research lab in 2019 now fits under a desk. We wanted to know exactly what that involves, at every step, and we wanted to be able to show it rather than describe it.

## Why an IT firm would do this

Most of the conversations we have with clients about AI come down to a version of the same question: what is actually inside the thing we are being asked to trust? Vendors answer that question with adjectives. We wanted to be able to answer it with specifics: what the model was trained on and under what license, what it cost in hardware and hours, what it can and cannot do and how that was measured, and what it looks like when it fails.

The only way to answer those with confidence is to have built one. So we did, and we wrote down everything.

## What Elroy is

A 360-million-parameter model, trained on about 20 billion tokens of text, roughly two thirds Python source code and one third English. It uses the same architecture as the open models released over the last two years, at a fraction of the size. After training it was taught to answer questions in a chat format, so you can ask it to write a function or explain what a dictionary is, and it will.

Its scores on the standard coding tests are 27% on HumanEval and 39% on MBPP. For comparison, the first version of GitHub Copilot in 2021 scored about 29% on the first of those with a model 33 times larger. Elroy is honest about its limits because we wrote its limits into its training data.

## What it cost

Two NVIDIA RTX A4500 cards, which are workstation cards you can buy retail, about six days of continuous training, and roughly 160 gigabytes of disk. The electricity was about $21 at our 27 cents per kilowatt-hour. No cloud, no API bills, no data we did not have the right to use. Every dataset is public and permissively licensed, and the model card lists each one.

## What we learned

Three things stood out, and each gets its own post over the coming weeks.

The data is where the risk lives. Two of the four public datasets we used had problems that a careful look found and a careless build would have shipped. One of them would have taught the model to start every program with a line of garbage. When a vendor tells you their model was trained on "high-quality data," that phrase is doing a great deal of work.

The numbers on the box are not the numbers you get. Our cards were rated at 94 trillion operations per second; they deliver 78 on a clean test and about 55 in real training. Every plan we made from the spec sheet was wrong until we measured.

Small models are useful when they are narrow. Elroy is bad at most things and reasonable at one. That is the shape most useful business AI will take: not one model that does everything, but small, inspectable ones that do a specific job under your control.

## Where to find it

The code, the nine-chapter write-up and the training logs are at github.com/strifero/elroy. The model weights are on Hugging Face as elroy-350m-base and elroy-350m-chat, under the Apache 2.0 license. Anyone with a single decent graphics card can follow the chapters and train the smaller configuration in an afternoon.

If you are trying to work out what AI should and should not be doing inside your firm, this is the kind of ground-level understanding we bring to that conversation. Get in touch.
