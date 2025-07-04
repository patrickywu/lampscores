# The Language Model Pairwise Comparison Framework

Last updated: July 3, 2025

Authors: Patrick Y. Wu, Jonathan Nagler, Joshua A. Tucker, and Solomon Messing

**TL;DR**: This is a two-part application that implements the **La**nguage **M**odel **P**airwise Comparison (LaMP) framework. The goal this framework is to estimate the latent positions of politicians along a specified political or policy dimension using pairwise comparisons with an LLM.

## Shiny Applications
The LaMP framework is available as a two-part application.

* [Generating Pairwise Comparisons Using an LLM](https://0197d1ff-abb0-a90f-ee26-39456c1b3378.share.connect.posit.cloud/): A Python-based Shiny app that uses pairwise comparison prompts with an LLM to compare politicians along a specified dimension. An API key is required. It is also available as a Python module.
* [Calculating Latent Positions of Politicians Using Bradley-Terry](https://0197d200-764b-fbc8-9714-b2877ee9e77c.share.connect.posit.cloud/): An R-based Shiny app that takes as input the output of the Python application and estimates the latent positions of politicians using the Bradley-Terry model.

## Usage in Python
To use the LaMPscores class in Python, you can install our package using the following.

```
pip install "lampscores@git+https://github.com/patrickywu/lampscores.git"
```

An example of how to use the LaMPscores class is below.

```python
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

## Demonstration Video
A video demonstration the application can be found [here](https://youtu.be/ojS_g4SeXFE).