# Top-K Movie Recommendation — MovieLens (Traditional RecSys + RAG)

This project was part of the **Advanced Topics in Data Science** course at FCUP Porto, completed by the three students Nina Lichtenberger, Aleksandra Rodzinka and Mija Aneta Stasiulionytė.

The task is top-K movie recommendation using implicit feedback (ratings ≥ 4) on the **MovieLens ml-latest-small** dataset (610 users, 9,724 rated movies). The project is split into two legs: a comparison of four classical recommender-system approaches, and a Retrieval-Augmented Generation (RAG) pipeline that exploits movie content (plot, genres, keywords) via an LLM, evaluated against the best traditional model.

Success is measured with **Precision@3, Recall@3, and F1@3**, benchmarked against a non-personalised popularity baseline.

## Project pipeline

**1 — Business & data understanding**
- Formulated as two ML problems: explicit rating prediction and top-K ranking
- The rating matrix is very sparse; user activity and movie popularity are both highly uneven
- Basket size and movie popularity are heavy-tailed but formally reject both a power-law and lognormal fit (likelihood-ratio test)

**2 — Data transformation**
- Users with fewer than 5 positive interactions were excluded to ensure sufficient signal and allow chronological splitting
- Explicit (rating-based) and implicit (binary, rating ≥ 4) versions of the data were created
- Chronological 70/15/15 train/validation/test split to prevent temporal leakage
- A representative 20% probe sample (validated via two-sample KS test, KS = 0.05, p = 0.925) was used for the initial hyperparameter search, with the full dataset used to confirm final results

**3 — Modelling on the probe sample**
Four recommender approaches were tuned and compared against the popularity baseline:
- **Association Rules (FP-Growth)**: grid search over 56 (min_support × min_confidence) combinations; best config (ms=0.04, mc=0.60) reached F1@3 = 0.0357
- **Collaborative Filtering (cosine kNN)**: user-based clearly outperformed item-based; best explicit user-based CF (5 neighbours) reached F1@3 = 0.0610, the strongest probe result
- **Matrix Factorization (MFPQ)**: random search over factors/learning rate/epochs/regularization; best explicit config (50 factors, λ=0.001, lr=0.001, 10 epochs) reached F1@3 = 0.0527
- **Bayesian Personalized Ranking (BPR)**: grid search over 256 combinations; best config (50 factors, lr=0.05, 50 iterations, λ=0.1) reached F1@3 = 0.0442, a 3× improvement over baseline
- **Neural Collaborative Filtering (NCF)**: random search over 500 trials per mode (implicit/explicit); best F1@3 = 0.0404 (implicit) / 0.0392 (explicit) — consistently the weakest model and dropped from further testing, although it was the most sophisticated model

**4 — Modelling on the full dataset**
Representative configurations from the probe stage were re-evaluated on the full dataset. Rather than repeating the full search, the best configurations as well as a random and a markedly different configuration (when compared to the best one) were evaluated to ensure an extensive search space. The following were the best results for the full dataset:

| Model | P@3 | R@3 | F1@3 |
|---|---|---|---|
| AR (ms=0.04, mc=0.60) | 0.0746 | 0.0322 | **0.0450** |
| CF (Explicit user-kNN, k=40) | 0.1150 | 0.0279 | **0.0449** |
| MFPQ (Binary, 50f, lr=0.01, reg=0.001, ep=40) | 0.0719 | 0.0314 | **0.0437** |
| BPR (100f, lr=0.01, iter=100, reg=0.001) | 0.0630 | 0.0314 | **0.0419** |
| Popularity baseline | 0.0337 | 0.0183 | **0.0237** |

- Hyperparameters tuned on the probe sample did not always transfer cleanly to the full data (e.g. BPR's best full-data config used a very different region of hyperparameter space than the probe-tuned one; CF's optimal neighbourhood grew from k=5 on the probe to k=40 on the full data)
- AR and CF tied on F1@3; **CF was selected as the final model** for its much higher precision (0.1150 vs. 0.0746), which matters most when only 3 recommendations are shown to the user
- All four models beat the popularity baseline by 80–90%

**5 — Held-out test set evaluation (best model: explicit user-based CF, k=40)**

| K | Precision | Recall | F1 |
|---|---|---|---|
| 3 | 0.0735 | 0.0158 | 0.0260 |
| 5 | 0.0703 | 0.0264 | 0.0384 |
| 10 | 0.0662 | 0.0485 | 0.0560 |

CF comfortably beats the popularity baseline at every K, confirming genuine predictive power on unseen data.

**6 — Leg 2: RAG recommender**
- Vector database built from movie metadata (TMDB overviews, genres, keywords) restricted to the 2,175 unique movies in the probe sample
- Two-stage LLM pipeline: (1) summarise a user's viewing history into a short preference query (sessions capped at 20 movies to fit the context window), (2) retrieve semantically similar movies and have the LLM select/explain the final recommendations
- **Small-scale (5-user) evaluation**: RAG outperformed CF at K=10 on precision, recall, and F1
- **Larger-scale (50-user) same-condition evaluation**: CF consistently outperformed the RAG retriever+reranker across all K — the 5-user result did not generalise, most likely due to small-sample variance
- **Qualitative evaluation**: automatic and human scoring of the LLM's explanations rated clarity highly (mean 4.85/5) and usefulness/faithfulness/specificity solidly (mean 3.90/5 each), with some explanations judged too generic

## Key takeaways
- Explicit user-based collaborative filtering with cosine similarity was the strongest and most robust traditional model, outperforming matrix factorization, BPR, and neural CF
- Association Rules remain competitive on raw F1@3 and offer a unique advantage: interpretable, human-readable rules
- Hyperparameters tuned on a data subsample do not always transfer to the full dataset — several models needed re-tuning or showed different optimal regions at scale
- A content-based RAG pipeline can match or exceed CF in small samples, but did not outperform CF in a larger, controlled comparison; its main value lies in generating clear, human-readable explanations rather than raw ranking accuracy

## Running the notebook
The notebook must be placed in the **same directory** as the files in the `files_to_run` folder to run correctly and make use of cached results (e.g. cached hyperparameter search results and precomputed embeddings/retrieval indices).

## Repository structure

| Folder / File | Contents |
|---|---|
| `TACD_project.ipynb` | Notebook including explanations and analysis (ipynb) |
| `files_to_run/` | Supporting data and cached result files required to run the notebook |
