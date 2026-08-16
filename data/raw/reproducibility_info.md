# Reproducibility Information

This document describes the technical setup and conditions under which 
the writing process and the embedded sycophancy experiment were conducted, 
to support reproducibility and interpretation of the accompanying data.

## Model 
- **Model:** GPT-5.5 Thinking
- **Mental effort:** High
- **Interface:** ChatGPT Desktop App (Windows 11 Pro)
- **Subscription tier:** ChatGPT Plus
- **Memory function:** Disabled for the duration of the study, to prevent carry-over of information between chat sessions and ensure that all context was provided explicitly via the context prompt
- **Auto-Update:** Disabled 
- **Model parameters (e.g., temperature, top-p):** default settings applied by OpenAI at the time of the experiment were used throughout. 
  No parameter-level configuration was performed.
- **App-Version:** 1.2026.119
- **Prompting language:** German for the writing process, English for the sycophancy experiment

## Interface
- **Language:** automatical detection
- **Custom Instructions:** None
- **Projecthints:** Yes (see context prompt)
- **Memory:** Disabled 
- **Web Search:** Enabled
- **Improve Model:** Disabled 
- **One Chat per Step:** Yes, to separate contexts, each with an opening system prompt

## Study Period
- **Writing process (GPT-5.5 run):** May 2026 
- **Embedded sycophancy experiment:** conducted as part of Step 4 (Study Design & Data Collection), within the same period

## Session Structure
- A separate chat session was initiated for each of the twelve writing steps (S1–S12), each preceded by a specific system prompt (recorded per step in `process_diary.csv` under `System Prompt`), followed by task-specific prompts (see `prompt_archive.csv`)
- A persistent context prompt provided background information (paper topic, target conference, formatting requirements, summary of completed steps) across sessions
- **Prompting-Structure:** context prompt → system prompt → task-specific prompt(s) → AI output → human intervention (if any) → next task-specific prompt (if any) → AI output → human intervention (if any) → … until step completion
- **Documentation:** Process diary, prompt archive and hallucination log to record all prompts, interventions and identified factual errors
- **Time tracking:** Timer-App - AI-Time (min) vs. Human-Time (min) logged per step