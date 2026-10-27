# Deployment checklist

These provider spend controls must be set by an account owner in the provider
dashboard. The checkboxes are intentionally unchecked; this repository cannot
inspect or change account billing settings.

- [ ] **OpenRouter** — Set a per-key spending limit with a daily reset in the
  dashboard under **Settings → Security → Key safety**. The key owner can set
  a limit that blocks requests after it is reached. See [OpenRouter Security
  Center](https://openrouter.ai/blog/announcements/security-center/) and
  [spend controls](https://openrouter.ai/blog/tutorials/team-spend-controls-setup/).
- [ ] **Gemini API** — Gemini does not document a daily spend cap. Set the
  project-level **monthly spend cap** in **Google AI Studio → Spend → Monthly
  spend cap → Edit spend cap**; inspect rate quotas in **Dashboard → Rate
  Limits**. Google describes the project cap as experimental and notes that
  usage can exceed it during billing latency. See [Gemini API billing and
  spend caps](https://ai.google.dev/gemini-api/docs/billing/) and [rate
  limits](https://ai.google.dev/gemini-api/docs/rate-limits).
