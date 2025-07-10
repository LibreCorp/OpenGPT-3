@krll-corp I don't recommend merging this branch with master. It has some logical flows that I'd like to address first.

what I propose:
- not all datasets are made equal. I propose creating a system that will grab .yaml file where all configs will be stored (currently this is a job of TrainingSpec/EvaluationSpec/GenerationSpec).
- switch to datasets library instead of downloading data files (in progress, partially done).
- refactor train pipeline and remove specs
- make model architecture compatible with huggingface by default