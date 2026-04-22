As part of the University of California, San Francisco [Abbasi Laboratory]([url](https://abbasilab.org/)), Hanav Modasiya (intern), Dr. Patrick Xian, and Dr. Reza Abbasi-Asl developed a framework for implementing stochastic behavioral interventions on multi-turn LLM agents.

The paper is currently under review, but the premise is that we developed an agent that recursively determines N failure points in a multi-turn LLM agent's reasoning trajectory and plans N behavioral interventions to combat those failure points. 

This is the implementation of stochastic behavioral interventions on the [Craft-MD]([url](https://rajpurkarlab.github.io/craft-md-pages/)) benchmark, specifically its "Infectious Disease" and "Neurology" categories. The environment is forked from the the Craft-MD repository, and the primary code for behavioral interventions are in . 

Through our evaluations, the following results were achieved: (with more evaluations on other models coming soon)

### Pass Rate before and after behavioral interventions

| Worker / User Agent  | Intervenor         | Craft-MD Infectious Disease   | Craft-MD Neurology   |
| -------------------- | ------------------ | ----------------------------- | -------------------- |
| GPT-4.1-mini         | None               | 25.9                          | 25.0                 |
| GPT-4.1-mini         | GPT-5-mini (Bo5)   | 44.4 (+71.4%)                 | 38.9 (+55.6%)        |
| -------------------- | ------------------ | ----------------------------- | -------------------- |
| GPT-5-mini           | None               | 42.4                          | 39.0                 |
| GPT-5-mini           | GPT-5-mini (Bo5)   | 57.6 (+35.7%)                 | 58.5 (+50.0%)        |

