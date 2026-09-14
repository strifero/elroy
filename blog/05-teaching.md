# Teaching a model to answer, not just continue

*Draft 5 for strifetech.com. Week five. Links to Chapter 7. Needs before/after samples.*

There is a step in building a language model that most people never hear about, and it is the step that turns a statistical curiosity into something you can talk to. It took us a few hours on one graphics card. It is the reason a model can say who it is.

## What a freshly trained model actually does

After six days of training, Elroy could predict the next word of Python or English text with real skill. Ask it, in a file, to define a function that checks whether a number is prime, and it would write the function. But ask it the same thing as a question, in plain English, and it would not answer. It would continue. It would write the next sentence that tends to follow such a question on the internet: a rephrasing, a list of requirements, sometimes the beginning of an answer, and then it would keep going, forever, because nothing had ever taught it that answers end.

Before this step:

> [sample: base model continuing a question instead of answering it]

This is what every large model looks like before its makers do the next part. The knowledge is in there. The behaviour is not.

## The fix is a format

We showed Elroy about 80,000 examples of programming questions and answers, laid out in one fixed pattern: a system line, a user question, an assistant reply, and a special marker that means "the reply is finished." Then we trained it for two passes over that data at a gentle learning rate, only on the reply portion, never on the question. That last detail matters: we want the model to learn how to answer, not how to ask.

To that public data we added 45 examples we wrote ourselves. Who Elroy is. What it cannot do. That it should say so. A dozen beginner concepts explained the way we would explain them. Those 45 examples, repeated a few times, are the entire source of the model's personality and its honesty about its limits. Every model you use had a file like that, written by people whose names you do not know.

After:

> [sample: chat model answering the same question and stopping]

## Why this matters to a firm evaluating AI

Two things. First, the "assistant" behaviour is a thin layer over a text predictor. When a model tells you it cannot do something, or that it is not sure, that is not introspection; it is a pattern from a file like ours. It is only as reliable as the examples were. Second, that layer is where a vendor's choices about tone, refusals and what the model claims about itself live, and it is invisible to you. You can ask to see it. Most vendors will not show you. Ours is 45 lines in a public repository.

Elroy, asked who made it, says Strife Technologies built it as a teaching project. It says that because we wrote it down and showed it to the model four times. Everything else it says about itself has the same provenance.

The data, the template and the training details are in Chapter 7.
