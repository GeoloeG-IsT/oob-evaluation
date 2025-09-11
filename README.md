# oob-evaluation-claude

## Commands

```bash
uvx --from git+https://github.com/github/spec-kit.git specify init --here --ai claude
find . -type f -name "*.sh" -exec chmod +x {} +
```

## Claude prompts (Inside Claude Code)

### Specify

- /specify @SPEC.md
- "FR-016 -> models which should be supported are: YOLOE, YOLO11,   │
│   YOLO12, SAM 2, RT-DETR                                           │
│   For each of those, finf out which variant exist (small, medium,  │
│   large, ...)                                                      │
│                                                                    │
│   FR-017 -> find out which variant exist for SAM 2                 │
│                                                                    │
│   FR-018 -> no max                                                 │
│                                                                    │
│   FR-019 -> execution time and mAP (any other relevant for object  │
│   detection ?)                                                     │
│                                                                    │
│   FR-020 -> indefinitely                                           │
│                                                                    │
│   FR-021 -> all image formats (I'll use big TIFF from satellite    │
│   for test)                                                        │
│                                                                    │
│   FR-022 -> no max size                                            │
│                                                                    │
│   FR-023 -> no auth for user                                       │
│                                                                    │
│   FR-024 -> no control access needed"
- Can you review the requirements, correct any ambiguities and add any missing ones?
- Any interesting missing features?
- Add model deployment

### Plan

- /plan @PLAN.md
- "Now I want you to go and audit the implementation plan and the implementation detail files.
Read through it with an eye on determining whether or not there is a sequence of tasks that you need
to be doing that are obvious from reading this. Because I don't know if there's enough here. For example,
when I look at the core implementation, it would be useful to reference the appropriate places in the implementation
details where it can find the information as it walks through each step in the core implementation or in the refinement."
- yes, please, go ahead

### Tasks

- /tasks
- Don't forget to set a venv before installing the python dependencies
- Add a .gitignore file with all .env and *.pyc files
