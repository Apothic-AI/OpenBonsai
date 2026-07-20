# Contributing to OpenBonsai

OpenBonsai welcomes independently reproducible low-bit experiments. A useful
contribution either adds evidence, implements a discriminating baseline, or
closes a reproducibility gap.

Please include:

- the hypothesis tested and a predicted outcome written before the run;
- exact model/tokenizer revisions and data manifests;
- actual hard-export artifacts, sizes, and precision exceptions;
- fixed prompts, decoding settings, seeds, and scoring code;
- conversion tokens, accelerator hours, and approximate FLOPs;
- negative results as well as favorable metrics.

Do not describe a method as recovered from PrismML without direct evidence. Do
not add leaked, non-public, or improperly licensed material. Keep comparisons
against the exact same base model and include naïve RTN as a control.

Run the dependency-light suite before submitting changes:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

