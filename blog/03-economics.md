# What it costs to train a model at home, in measured numbers

*Draft 3 for strifetech.com. Week three. Links to Chapters 0 and 4.*

There is a widely held belief that training a language model requires a data center. For the models in the news, that is true. For a model the size of the one that made the news in 2019, it is not, and we can put numbers on it because we just did it.

## The hardware

Two NVIDIA RTX A4500 graphics cards. These are workstation cards, not gaming cards and not data-center cards; they were about $1,000 each when we bought them and sit in an ordinary tower with a consumer processor and 32 gigabytes of memory. The rest of the machine is unremarkable. It also ran other workloads for months before this project, and had to be cleaned up first.

## The spec sheet versus the meter

Our first plan estimated nine days of training. It was built on the number printed on the box: each card is rated at 94 trillion operations per second. The first thing we did was measure. A clean test of the cards' core operation returned 78. In real training, where the cards also have to move data and wait for each other, the sustained rate was 55.

The plan then went the other direction. We had assumed real training would reach 35% of the rated figure, which is a conservative rule of thumb. After tuning the batch size on the actual cards and letting the compiler fuse the small operations, we reached 70% of the measured figure. The nine-day estimate became six. Neither correction came from the documentation. Both came from running the thing and reading the numbers.

This is the same discipline we apply to anything we run for clients. Vendor figures are the starting point for a measurement, not a substitute for one.

## The run

Six days of continuous training, 38,146 update steps, each consuming about half a million tokens of text. A checkpoint saved every 55 minutes so that a crash costs an hour, not a week. Roughly 160 gigabytes of disk for the data and the saved states. The two cards draw about 400 watts under load and the rest of the machine about 150, so the run consumed roughly 79 kilowatt-hours over its 143 hours, about $21 at our rate of 27 cents. The cards alone were $15 of that. We did not put a meter on it; those are rated draws at the 100% utilization the cards reported the whole time.

For scale: the first version of the model this compares to, GPT-2 in 2019, was trained on a cluster of data-center accelerators at a company with a nine-figure budget. Seven years later, a somewhat smaller but better-trained equivalent is a week on a workstation. That curve is the single most important fact about where this technology is going, and it is why "AI" will not stay a thing you can only rent.

## What it does not cost

No cloud account. No API key. No per-token bill. No data leaving the building. No license terms that change under you. The model is ours, the data provenance is documented, and it runs on the same card that trained it, or on a laptop with a smaller configuration.

That is not the right trade for every use. For a general assistant, renting a frontier model is cheaper and far more capable. For a narrow, repetitive task on sensitive data inside a professional-services firm, the calculation looks different, and it is one we can now do with real numbers rather than estimates.

The full numbers, including the ones that were wrong, are in Chapters 0 and 4 of the write-up.
