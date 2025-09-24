# Answers

1) If you only had 200 labeled replies, how would you improve the model without collecting thousands more?
I’d fine‑tune a small, pretrained transformer with strong regularization, early stopping, and class‑balanced sampling. To squeeze more signal, I’d add light data augmentation (paraphrases) and semi‑supervised learning with high‑confidence pseudo‑labels from a larger unlabeled pool, plus domain‑adaptive pretraining on unlabeled in‑domain text.

2) How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?
Start with curated, audited data and track slice‑based metrics (e.g., by geography, gender proxies) to catch drift and bias. In production, enforce confidence thresholds with human review for low‑confidence cases, apply content filters/policy rules, and keep a feedback loop for periodic, documented re‑training.

3) Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non‑generic?
Give the model structured context (role, industry, pain point, recent news) and a few concise, style‑consistent examples. Add explicit constraints—length, tone, banned clichés—and require the opener to cite which provided facts it used; finish with a brief self‑check step to reject generic or ungrounded outputs.

