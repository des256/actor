- 3x LLM inference is slower on Jetson (thinking time around 3s)... On desktop (RTX4080 SUPER) it's around 500ms

  --> A solution might be to parallelize intent analysis next to the main LLM, but I'm not sure that will help on the Jetson...

- history summarizer, but only call it in spare time
- high level steering system
- proactive turn taking (robot speaks first)
- XML-style inline tagged animation
- audio-to-viseme
- speaker diarization
- implement VLM as main LLM
- Qwen3 2.5 for main LLM (try)
- clean up st7789.rs
