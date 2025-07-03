# The LaMP Scores Framework

Last updated: July 3, 2025

Authors: Patrick Y. Wu, Jonathan Nagler, Joshua A. Tucker, and Solomon Messing

This repository includes an implementation of the LaMP scores framework.

## Shiny Applications
The LaMP framework is available as a two-part application.

* [Generating Pairwise Comparisons Using an LLM](https://0197d1ff-abb0-a90f-ee26-39456c1b3378.share.connect.posit.cloud/): This Python-based app creates pairwise comparison prompts and inputs them to an LLM of your choice. An API key is required.
* [Calculating Latent Positions of Politicians Using Bradley-Terry](https://0197d200-764b-fbc8-9714-b2877ee9e77c.share.connect.posit.cloud/): This R-based app takes as input the output of the previous Python app and estimates the latent positions of politicians using the Bradley-Terry model

## Usage in Python
To use the LaMPscores class in Python, you can install our package using the following.

```
pip install "lampscores@git+https://github.com/patrickywu/lampscores.git"
```

An example of how to use the LaMPscores class is below.

```
from openai import AsyncOpenAI
from lampscores import LaMPscores

lamp_ideology = LaMPscores(client=client,
                           model="gpt-4.1-nano",
                           congress_number=116,
                           chamber="S",
                           politician_type="senator",
                           unidirectional=False,
                           liberal_direction_prompt=None, # defaults to a built-in ideology prompt asking which senator is more liberal
                           conservative_direction_prompt=None, # defaults to a built-in ideology prompt asking which senator is more conservative
                           randomize_pairwise_order_seed=42,
                           concurrency=140, # number of API requests in-flight
                           sample_per_item=10,
                           temperature=1.0,
                           top_p=1.0)

await lamp_ideology.run()

lamp_ideology.matchup_results_df # Use this to see the resulting pairwise comparisons
```