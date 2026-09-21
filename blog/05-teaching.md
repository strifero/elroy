# Teaching a model to answer, not just continue

*Draft 5 for strifetech.com. Week five. Links to Chapter 7. Needs before/after samples.*

There is a step in building a language model that most people never hear about, and it is the step that turns a statistical curiosity into something you can talk to. It took us 93 minutes on one graphics card. It is the reason a model can say who it is.

## What a freshly trained model actually does

After six days of training, Elroy could predict the next word of Python or English text with real skill. Ask it, in a file, to define a function that checks whether a number is prime, and it would write the function. But ask it the same thing as a question, in plain English, and it would not answer. It would continue. It would write the next sentence that tends to follow such a question on the internet: a rephrasing, a list of requirements, sometimes the beginning of an answer, and then it would keep going, forever, because nothing had ever taught it that answers end.

Before this step, we typed "Write a Python function that checks whether a number is prime." and got, in full:

```
# Note: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
```

A comment. It read the request as the top of a file and produced the line that most often comes next in such files, then stopped, because that is what the text it learned from does. Asked the same thing as a function signature instead of a sentence, it wrote the function perfectly. The knowledge was there; the manners were not.

This is what every large model looks like before its makers do the next part. The knowledge is in there. The behaviour is not.

## The fix is a format

We showed Elroy about 82,000 examples of programming questions and answers, laid out in one fixed pattern: a system line, a user question, an assistant reply, and a special marker that means "the reply is finished." Then we trained it for two passes over that data at a gentle learning rate, only on the reply portion, never on the question. That last detail matters: we want the model to learn how to answer, not how to ask.

To that public data we added 45 examples we wrote ourselves. Who Elroy is. What it cannot do. That it should say so. A dozen beginner concepts explained the way we would explain them. Those 45 examples, repeated a few times, are the entire source of the model's personality and its honesty about its limits. Every model you use had a file like that, written by people whose names you do not know.

After, the same request:

> Here is a simple Python function that checks whether a number is prime:
>
> (a correct twelve-line function)
>
> This function works by first checking if the number is less than or equal to 1, in which case it is not prime. Then it checks if the number is less than or equal to 3, in which case it is prime. If the number is divisible by 2 or 3, it is not prime. If the number is divisible by any number up to its square root, it is prime.

A function, an explanation, and a full stop. The last sentence of the explanation is wrong (divisible means not prime), which is the model in a nutshell: the format is learned, the content is what the size allows.

## Why this matters to a firm evaluating AI

Two things. First, the "assistant" behaviour is a thin layer over a text predictor. When a model tells you it cannot do something, or that it is not sure, that is not introspection; it is a pattern from a file like ours. It is only as reliable as the examples were. Second, that layer is where a vendor's choices about tone, refusals and what the model claims about itself live, and it is invisible to you. You can ask to see it. Most vendors will not show you. Ours is 45 lines in a public repository.

Elroy, asked who made it, says Strife Technologies built it as a teaching project. It says that because we wrote it down and showed it to the model four times. Everything else it says about itself has the same provenance.

The data, the template and the training details are in Chapter 7.
