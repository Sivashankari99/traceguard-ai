from pathlib import Path
from collections import defaultdict, deque
import ast
import json
import math
import re

import numpy as np
import pandas as pd
from minsearch import Index

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required. Install project dependencies first."
    ) from exc


class TraceGuard:
    """Hybrid lexical + semantic + traceability-aware engineering impact analyzer."""

    IMPORTANT_TYPES = [
        "Change Request", "Problem Report", "ALM Input", "ALM Requirement",
        "ALM Specification", "ALM Test Suite", "ALM Test Case", "Task", "Release",
    ]
    DOCUMENT_COLUMNS = [
        "ID", "Document_ID", "Type", "Summary", "Text",
        "State", "Project", "Spawns", "Covers", "Search_Text",
    ]
    CHANGE_TYPES = {"Change Request", "Problem Report"}
    ALM_BASELINE_TYPES = {
        "ALM Input", "ALM Requirement", "ALM Specification",
        "ALM Test Suite", "ALM Test Case",
    }
    BASELINE_IMPACT_LEVELS = {"High", "Medium"}

    def __init__(
        self,
        data_path=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        per_type_candidates=None,
        llm_context_limit=80,
        unlinked_context_fraction=0.25,
        max_traceability_hops=4,
        llm_model="gpt-4o-mini",
    ):
        if data_path is None:
            # Works when src/ is directly under the repository root.
            data_path = Path(__file__).resolve().parents[1] / "data"
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.llm_model = llm_model
        self.llm_context_limit = llm_context_limit
        self.unlinked_context_fraction = unlinked_context_fraction
        self.max_traceability_hops = max_traceability_hops
        self.default_per_type_candidates = 50
        self.per_type_candidates = per_type_candidates or {
            t: 50 for t in self.IMPORTANT_TYPES
        }

        self.traceability_seed_mode = "rank"
        self.traceability_seed_max_rank = {
            "Change Request": 20,
            "Problem Report": 20,
        }
        self.traceability_seed_min_similarity = {
            "Change Request": None,
            "Problem Report": None,
        }

        self._load_and_prepare_data()
        self._build_lexical_indexes()
        self._build_embeddings()
        self._build_traceability_graph()

    def _load_and_prepare_data(self):
        self.artifacts_df = pd.read_csv(self.data_path / "artifacts.csv")
        self.baselines_df = pd.read_csv(self.data_path / "baselines.csv")

        for col in ["Summary", "Text", "Spawns", "Covers", "State", "Project", "Document_ID"]:
            if col not in self.artifacts_df.columns:
                self.artifacts_df[col] = ""
            self.artifacts_df[col] = self.artifacts_df[col].fillna("").astype(str)

        self.artifacts_df["ID"] = self.artifacts_df["ID"].astype(str)
        self.artifacts_df["Type"] = self.artifacts_df["Type"].fillna("").astype(str)
        self.artifacts_df["Search_Text"] = (
            self.artifacts_df["ID"].str.strip() + " | " +
            self.artifacts_df["Type"].str.strip() + " | " +
            self.artifacts_df["Summary"].str.strip() + " | " +
            self.artifacts_df["Text"].str.strip()
        ).str.strip(" |")

        available_types = set(self.artifacts_df["Type"].unique())
        self.retrieval_types = [t for t in self.IMPORTANT_TYPES if t in available_types]
        self.known_ids = set(self.artifacts_df["ID"].astype(str))
        self.artifact_lookup = (
            self.artifacts_df.set_index("ID", drop=False).to_dict(orient="index")
        )

    def _build_lexical_indexes(self):
        self.lexical_indexes = {}
        self.documents_by_type = {}

        for artifact_type in self.retrieval_types:
            type_df = self.artifacts_df[self.artifacts_df["Type"] == artifact_type].copy()
            docs = type_df[self.DOCUMENT_COLUMNS].fillna("").to_dict(orient="records")
            if not docs:
                continue

            idx = Index(
                text_fields=["Search_Text"],
                keyword_fields=["ID", "Document_ID", "State", "Project", "Type"],
            )
            idx.fit(docs)
            self.lexical_indexes[artifact_type] = idx
            self.documents_by_type[artifact_type] = docs

    def _build_embeddings(self):
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_cache = {}

        for artifact_type in self.retrieval_types:
            type_df = self.artifacts_df[self.artifacts_df["Type"] == artifact_type].copy()
            texts = type_df["Search_Text"].tolist()
            if not texts:
                continue

            embeddings = self.embedding_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.embedding_cache[artifact_type] = {
                "ids": type_df["ID"].tolist(),
                "embeddings": np.asarray(embeddings),
            }

    @staticmethod
    def _parse_link_ids(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        parts = re.split(r"[,;\n|]+", text)
        return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]

    def _build_traceability_graph(self):
        self.graph = defaultdict(list)

        def add_edge(src, dst, relationship):
            if src in self.known_ids and dst in self.known_ids:
                self.graph[src].append((dst, relationship))
                self.graph[dst].append((src, f"reverse:{relationship}"))

        for _, row in self.artifacts_df.iterrows():
            src = str(row["ID"])
            for dst in self._parse_link_ids(row.get("Spawns", "")):
                add_edge(src, dst, "Spawns")
            for dst in self._parse_link_ids(row.get("Covers", "")):
                add_edge(src, dst, "Covers")

    def _lexical_retrieve(self, query_text, artifact_type, limit):
        idx = self.lexical_indexes.get(artifact_type)
        if idx is None:
            return []
        results = idx.search(
            query=query_text,
            num_results=min(limit, len(self.documents_by_type[artifact_type])),
        )
        output = []
        for rank, item in enumerate(results, start=1):
            row = dict(item)
            raw_score = row.get("_score", row.get("score", np.nan))
            row["lexical_score"] = float(raw_score) if pd.notna(raw_score) else np.nan
            row["lexical_rank"] = rank
            row["retrieval_source"] = "Lexical retrieval"
            output.append(row)
        return output

    def _score_all_semantic(self, query_text, artifact_type):
        cached = self.embedding_cache.get(artifact_type)
        if cached is None:
            return pd.DataFrame()

        query_embedding = self.embedding_model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        similarities = cached["embeddings"] @ query_embedding
        order = np.argsort(-similarities)

        rows = []
        for rank, pos in enumerate(order, start=1):
            artifact_id = str(cached["ids"][pos])
            row = dict(self.artifact_lookup[artifact_id])
            row["semantic_similarity"] = float(similarities[pos])
            row["semantic_rank"] = rank
            rows.append(row)
        return pd.DataFrame(rows)

    def _add_candidate_evidence(self, store, row, source, **evidence):
        artifact_id = str(row["ID"])
        if artifact_id not in store:
            base = {col: row.get(col, "") for col in self.DOCUMENT_COLUMNS}
            base.update({
                "retrieval_sources": set(),
                "traceability_paths": [],
                "traceability_relationships": set(),
                "semantic_similarity": np.nan,
                "lexical_score": np.nan,
                "semantic_rank": np.nan,
                "lexical_rank": np.nan,
                "traceability_distance": np.nan,
            })
            store[artifact_id] = base

        item = store[artifact_id]
        item["retrieval_sources"].add(source)

        for key, value in evidence.items():
            if key in {"semantic_similarity", "lexical_score"}:
                if value is not None and pd.notna(value):
                    current = item.get(key, np.nan)
                    item[key] = float(value) if pd.isna(current) else max(float(current), float(value))
            elif key in {"semantic_rank", "lexical_rank"}:
                if value is not None and pd.notna(value):
                    current = item.get(key, np.nan)
                    item[key] = int(value) if pd.isna(current) else min(int(current), int(value))
            else:
                item[key] = value

    def _is_traceability_seed(self, row):
        artifact_type = row.get("Type")
        if artifact_type not in self.CHANGE_TYPES:
            return False
        rank_limit = self.traceability_seed_max_rank.get(artifact_type)
        similarity_limit = self.traceability_seed_min_similarity.get(artifact_type)

        rank_ok = (
            rank_limit is not None and pd.notna(row.get("semantic_rank"))
            and int(row["semantic_rank"]) <= int(rank_limit)
        )
        similarity_ok = (
            similarity_limit is not None and pd.notna(row.get("semantic_similarity"))
            and float(row["semantic_similarity"]) >= float(similarity_limit)
        )
        if self.traceability_seed_mode == "rank":
            return rank_ok
        if self.traceability_seed_mode == "similarity":
            return similarity_ok
        if self.traceability_seed_mode == "rank_or_similarity":
            return rank_ok or similarity_ok
        raise ValueError(f"Unknown traceability seed mode: {self.traceability_seed_mode}")

    def _expand_traceability(self, seed_ids):
        discoveries = defaultdict(list)
        for seed in seed_ids:
            queue = deque([(seed, [seed], [], 0)])
            best_distance = {seed: 0}
            while queue:
                current, path, edge_evidence, distance = queue.popleft()
                if distance >= self.max_traceability_hops:
                    continue
                for neighbor, relationship in self.graph.get(current, []):
                    if neighbor in path:
                        continue
                    new_distance = distance + 1
                    new_path = path + [neighbor]
                    new_edges = edge_evidence + [{
                        "from": current, "to": neighbor, "relationship": relationship
                    }]
                    discoveries[neighbor].append({
                        "seed_change_id": seed,
                        "distance": new_distance,
                        "path": new_path,
                        "edges": new_edges,
                    })
                    if new_distance < best_distance.get(neighbor, math.inf):
                        best_distance[neighbor] = new_distance
                        queue.append((neighbor, new_path, new_edges, new_distance))
        return discoveries

    @staticmethod
    def _reciprocal_rank(rank):
        return 0.0 if pd.isna(rank) else 1.0 / float(rank)

    def _review_rank(self, row):
        semantic_rr = self._reciprocal_rank(row.get("semantic_rank", np.nan))
        lexical_rr = self._reciprocal_rank(row.get("lexical_rank", np.nan))
        trace_signal = 1.0 if row.get("traceability_status") == "Linked" else 0.0
        return semantic_rr + lexical_rr + 0.10 * trace_signal

    @staticmethod
    def _evidence_category(row):
        text_signal = row["has_semantic_evidence"] or row["has_lexical_evidence"]
        trace_signal = row["has_traceability_evidence"]
        if text_signal and trace_signal:
            return "Strong candidate — similarity + traceability"
        if text_signal and not trace_signal:
            return "Similar candidate — no traceability found"
        if trace_signal and not text_signal:
            return "Traceability candidate — weak textual similarity"
        return "Low-confidence candidate"

    @staticmethod
    def _format_trace_paths(paths, max_paths=10):
        return [{
            "path": " -> ".join(p["path"]),
            "edges": p["edges"],
            "distance": p["distance"],
            "seed_change_id": p["seed_change_id"],
        } for p in paths[:max_paths]]

    def _build_context_records(self, selected_candidates):
        records = []
        for _, row in selected_candidates.iterrows():
            records.append({
                "artifact_id": str(row["ID"]),
                "artifact_type": str(row["Type"]),
                "state": str(row.get("State", "")),
                "project": str(row.get("Project", "")),
                "summary": str(row.get("Summary", "")),
                "text": str(row.get("Text", "")),
                "semantic_similarity": None if pd.isna(row.get("semantic_similarity")) else round(float(row["semantic_similarity"]), 4),
                "semantic_rank": None if pd.isna(row.get("semantic_rank")) else int(row["semantic_rank"]),
                "lexical_score": None if pd.isna(row.get("lexical_score")) else round(float(row["lexical_score"]), 4),
                "lexical_rank": None if pd.isna(row.get("lexical_rank")) else int(row["lexical_rank"]),
                "retrieval_sources": row["retrieval_sources"],
                "traceability_status": row["traceability_status"],
                "traceability_hops": None if pd.isna(row.get("traceability_hops")) else int(row["traceability_hops"]),
                "traceability_paths": self._format_trace_paths(row["traceability_paths"]),
                "unlinked_relevant": bool(row["unlinked_relevant"]),
                "candidate_category": row["candidate_category"],
                "review_rank_score": round(float(row["review_rank_score"]), 6),
            })
        return records

    def _build_prompt(self, query, context_records):
        context_json = json.dumps(context_records, indent=2, ensure_ascii=False)
        return f"""
You are TraceGuard AI, an engineering change impact analysis reviewer.

PROPOSED CHANGE:
{query}

CANDIDATE ARTIFACTS:
{context_json}

The retrieval pipeline has already discovered these candidates using independent
per-artifact lexical retrieval, semantic similarity, and/or traceability expansion.
Your job is to ASSESS the supplied candidates, not discover new artifacts.

CRITICAL FIRST STEP — INPUT RELEVANCE:

Before performing ANY impact assessment, classify the PROPOSED CHANGE as
either "Relevant" or "Irrelevant".

A proposed change is Relevant only if its actual subject and intent concern
the engineering/system/product domain represented by the candidate artifacts.

Do NOT classify an input as Relevant merely because retrieved candidates have
some semantic or lexical similarity. Retrieval always returns candidates and
those candidates may be false matches.

For example, questions about education, classes, travel, food, personal
activities, entertainment, or other unrelated topics are Irrelevant.

If the input is Irrelevant:
- set "input_relevance" to "Irrelevant"
- return "assessments": []
- do not assess any candidate artifact
- do not infer an engineering interpretation of the input
- set "overall_assessment" to:
  "The input is unrelated to the available engineering artifacts. No impact analysis was performed."

Only if input_relevance is "Relevant" should you continue with the impact
assessment rules below.

Rules:
1. Use only artifact IDs and evidence present in CANDIDATE ARTIFACTS.
2. Never invent IDs, relationships, paths, scores, releases, tasks, or engineering facts.
3. Similarity and traceability are separate evidence signals; neither alone proves impact.
4. A high-similarity artifact with "No relevant link found" must not be discarded solely because it is unlinked.
5. Do not impose a hard cosine-similarity threshold.
6. Treat "No relevant link found" as absence of discovered traceability, NOT as evidence of irrelevance.
7. review_rank_score is ordering only; never interpret it as impact probability or confidence.
8. If you cite a traceability relationship/path, reproduce only a supplied path.
9. Confidence is your review confidence from the supplied evidence, not a calibrated probability.
10. Prefer recall for a RELEVANT proposed change: include plausible candidates,
    but clearly communicate uncertainty. This rule does not apply when the
    proposed change is classified as Irrelevant.
11. Return valid JSON only. No markdown fences and no prose outside the JSON.

Return this exact top-level structure:
{{
  "proposed_change": "...",
  "input_relevance": "Relevant | Irrelevant",
  "assessments": [
    {{
      "artifact_id": "...",
      "artifact_type": "...",
      "impact_level": "High | Medium | Low | Review",
      "candidate_category": "Strong candidate — similarity + traceability | Similar candidate — no traceability found | Traceability candidate — weak textual similarity | Low-confidence candidate",
      "reason": "...",
      "evidence": {{
        "semantic_similarity": null,
        "lexical_score": null,
        "retrieval_sources": [],
        "traceability_paths": []
      }},
      "traceability_status": "Linked | No relevant link found",
      "confidence": "High | Medium | Low"
    }}
  ],
  "overall_assessment": "..."
}}
""".strip()

    def _call_llm(self, prompt):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai package to run the LLM assessment.") from exc

        client = OpenAI()
        response = client.responses.create(model=self.llm_model, input=prompt)
        raw_output = response.output_text.strip()
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON. Raw output:\n{raw_output}") from exc

    def _validate_grounding(self, impact_assessment, context_records):
        context_ids = {r["artifact_id"] for r in context_records}
        context_by_id = {r["artifact_id"]: r for r in context_records}
        allowed_paths_by_id = defaultdict(set)
        allowed_path_edges_by_id = defaultdict(dict)

        for record in context_records:
            for p in record["traceability_paths"]:
                allowed_paths_by_id[record["artifact_id"]].add(p["path"])
                allowed_path_edges_by_id[record["artifact_id"]][p["path"]] = p.get("edges", [])

        assessment_ids = {str(x.get("artifact_id", "")) for x in impact_assessment.get("assessments", [])}
        unknown_ids = assessment_ids - context_ids
        invalid_paths, invalid_linked, invalid_source_edges = [], [], []

        for artifact_id, path_map in allowed_path_edges_by_id.items():
            for path_text, edges in path_map.items():
                for edge in edges:
                    src, dst = str(edge.get("from", "")), str(edge.get("to", ""))
                    rel = str(edge.get("relationship", ""))
                    if (dst, rel) not in set(self.graph.get(src, [])):
                        invalid_source_edges.append({"artifact_id": artifact_id, "path": path_text, "edge": edge})

        for item in impact_assessment.get("assessments", []):
            artifact_id = str(item.get("artifact_id", ""))
            evidence = item.get("evidence", {}) or {}
            for claimed in evidence.get("traceability_paths", []) or []:
                path_text = str(claimed.get("path", "")) if isinstance(claimed, dict) else str(claimed)
                if path_text and path_text not in allowed_paths_by_id.get(artifact_id, set()):
                    invalid_paths.append({"artifact_id": artifact_id, "claimed_path": path_text})

            llm_status = str(item.get("traceability_status", ""))
            source_status = context_by_id.get(artifact_id, {}).get("traceability_status")
            if llm_status == "Linked" and source_status != "Linked":
                invalid_linked.append({"artifact_id": artifact_id, "llm_status": llm_status, "source_status": source_status})

        return {
            "passed": not (unknown_ids or invalid_paths or invalid_linked or invalid_source_edges),
            "unknown_ids": sorted(unknown_ids),
            "invalid_relationship_claims": invalid_paths,
            "invalid_linked_claims": invalid_linked,
            "invalid_source_edges": invalid_source_edges,
        }

    def _determine_baselines(self, impact_assessment):
        assessment_by_id = {
            str(item.get("artifact_id", "")): item
            for item in impact_assessment.get("assessments", [])
        }
        impacted_alm_ids = {
            aid for aid, item in assessment_by_id.items()
            if str(item.get("artifact_type", "")) in self.ALM_BASELINE_TYPES
            and str(item.get("impact_level", "")) in self.BASELINE_IMPACT_LEVELS
        }
        relevant_change_ids = {
            aid for aid, item in assessment_by_id.items()
            if str(item.get("artifact_type", "")) in self.CHANGE_TYPES
            and str(item.get("impact_level", "")) in self.BASELINE_IMPACT_LEVELS
        }

        release_to_changes = {}
        for _, row in self.artifacts_df[self.artifacts_df["Type"] == "Release"].iterrows():
            release_to_changes[str(row["ID"])] = set(self._parse_link_ids(row["Covers"]))

        baseline_members = defaultdict(set)
        for _, row in self.baselines_df.iterrows():
            baseline_members[str(row["Release_ID"])].add(str(row["Artifact_ID"]))

        rows = []
        for release_id, covered_changes in release_to_changes.items():
            supporting = sorted(relevant_change_ids.intersection(covered_changes))
            if not supporting:
                continue
            members = sorted(impacted_alm_ids.intersection(baseline_members.get(release_id, set())))
            if not members:
                continue
            rows.append({
                "Release_ID": release_id,
                "Baseline_ID": f"BL-{release_id}",
                "Supporting_CR_PR_IDs": ", ".join(supporting),
                "Affected_ALM_Artifact_IDs": ", ".join(members),
                "Supporting_CR_PR_Count": len(supporting),
                "Affected_ALM_Count": len(members),
                "Determination": "Traceability-supported baseline impact",
            })

        affected_df = pd.DataFrame(rows)
        if affected_df.empty:
            determination = {
                "status": "Undetermined",
                "reason": (
                    "No assessed High/Medium CR or PR could be connected through an explicit "
                    "Release Covers relationship to a Release baseline that also contains "
                    "assessed High/Medium ALM artifacts."
                ),
                "affected_release_ids": [],
                "affected_baseline_ids": [],
            }
        else:
            affected_df = affected_df.sort_values(
                ["Supporting_CR_PR_Count", "Affected_ALM_Count"], ascending=False
            ).reset_index(drop=True)
            determination = {
                "status": "Determined",
                "reason": "Explicit CR/PR -> Release traceability plus ALM baseline membership.",
                "affected_release_ids": affected_df["Release_ID"].tolist(),
                "affected_baseline_ids": affected_df["Baseline_ID"].tolist(),
            }
        return affected_df, determination

    def analyze(self, query, call_llm=True):
        """Analyze one free-text incoming CR/change and return final + diagnostic outputs."""
        query = str(query).strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        all_semantic_scores_by_type = {}
        semantic_results_by_type = {}
        lexical_results_by_type = {}

        for artifact_type in self.retrieval_types:
            all_scored = self._score_all_semantic(query, artifact_type)
            all_semantic_scores_by_type[artifact_type] = all_scored
            retain_k = self.per_type_candidates.get(
                artifact_type, self.default_per_type_candidates
            )
            semantic_results_by_type[artifact_type] = (
                all_scored.head(retain_k)
                .assign(retrieval_source="Semantic similarity")
                .to_dict(orient="records")
            )
            lexical_results_by_type[artifact_type] = self._lexical_retrieve(
                query, artifact_type, retain_k
            )

        candidate_store = {}
        for results in lexical_results_by_type.values():
            for row in results:
                self._add_candidate_evidence(
                    candidate_store, row, "Lexical retrieval",
                    lexical_score=row.get("lexical_score"),
                    lexical_rank=row.get("lexical_rank"),
                )
        for results in semantic_results_by_type.values():
            for row in results:
                self._add_candidate_evidence(
                    candidate_store, row, "Semantic similarity",
                    semantic_similarity=row.get("semantic_similarity"),
                    semantic_rank=row.get("semantic_rank"),
                )

        seed_ids = {
            aid for aid, item in candidate_store.items()
            if item.get("Type") in self.CHANGE_TYPES and self._is_traceability_seed(item)
        }
        discoveries = self._expand_traceability(seed_ids)
        for artifact_id, paths in discoveries.items():
            if artifact_id not in self.artifact_lookup:
                continue
            row = self.artifact_lookup[artifact_id]
            self._add_candidate_evidence(
                candidate_store, row, "Traceability expansion",
                traceability_distance=min(p["distance"] for p in paths),
            )
            item = candidate_store[artifact_id]
            for p in paths:
                item["traceability_paths"].append(p)
                item["traceability_relationships"].update(
                    edge["relationship"] for edge in p["edges"]
                )

        candidate_rows = []
        for artifact_id, item in candidate_store.items():
            paths = item.get("traceability_paths", [])
            sources = set(item.get("retrieval_sources", set()))
            by_similarity = bool({"Semantic similarity", "Lexical retrieval"} & sources)
            by_traceability = "Traceability expansion" in sources
            row = dict(item)
            row["retrieval_sources"] = sorted(sources)
            row["traceability_relationships"] = sorted(item["traceability_relationships"])
            row["traceability_paths"] = paths
            row["traceability_hops"] = min((p["distance"] for p in paths), default=np.nan)
            row["traceability_status"] = "Linked" if by_traceability else "No relevant link found"
            row["discovered_by_similarity"] = by_similarity
            row["discovered_by_traceability"] = by_traceability
            row["unlinked_relevant"] = by_similarity and not by_traceability
            candidate_rows.append(row)

        candidates_df = pd.DataFrame(candidate_rows)
        if candidates_df.empty:
            return {
                "query": query,
                "impact_assessment": {"proposed_change": query, "assessments": [], "overall_assessment": "No candidates found."},
                "impact_report_df": pd.DataFrame(),
                "affected_baselines_df": pd.DataFrame(),
                "baseline_determination": {"status": "Undetermined", "reason": "No candidates found.", "affected_release_ids": [], "affected_baseline_ids": []},
                "validation": {"passed": True},
                "ranked_candidates_df": pd.DataFrame(),
                "selected_candidates_df": pd.DataFrame(),
                "retrieval_summary_df": pd.DataFrame(),
            }

        candidates_df["has_semantic_evidence"] = candidates_df["semantic_similarity"].notna()
        candidates_df["has_lexical_evidence"] = candidates_df["lexical_score"].notna()
        candidates_df["has_traceability_evidence"] = candidates_df["discovered_by_traceability"]
        candidates_df["review_rank_score"] = candidates_df.apply(self._review_rank, axis=1)
        candidates_df["candidate_category"] = candidates_df.apply(self._evidence_category, axis=1)

        ranked = candidates_df.sort_values(
            ["review_rank_score", "semantic_similarity"],
            ascending=[False, False], na_position="last"
        ).reset_index(drop=True)

        unlinked_budget = min(
            int(round(self.llm_context_limit * self.unlinked_context_fraction)),
            int(ranked["unlinked_relevant"].sum()),
        )
        unlinked_selected = ranked[ranked["unlinked_relevant"]].head(unlinked_budget).copy()
        selected_ids = set(unlinked_selected["ID"].astype(str))
        remaining_budget = self.llm_context_limit - len(unlinked_selected)
        selected_parts = [unlinked_selected] if not unlinked_selected.empty else []

        if self.retrieval_types and remaining_budget > 0:
            per_type_budget = max(1, remaining_budget // len(self.retrieval_types))
            for artifact_type in self.retrieval_types:
                part = ranked[
                    (ranked["Type"] == artifact_type)
                    & (~ranked["ID"].astype(str).isin(selected_ids))
                ].head(per_type_budget)
                selected_parts.append(part)
                selected_ids.update(part["ID"].astype(str))

        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=ranked.columns)
        remaining = self.llm_context_limit - len(selected)
        if remaining > 0:
            already = set(selected["ID"].astype(str))
            selected = pd.concat(
                [selected, ranked[~ranked["ID"].astype(str).isin(already)].head(remaining)],
                ignore_index=True,
            )
        selected = (
            selected.drop_duplicates(subset=["ID"])
            .sort_values("review_rank_score", ascending=False)
            .head(self.llm_context_limit)
            .reset_index(drop=True)
        )

        context_records = self._build_context_records(selected)
        prompt = self._build_prompt(query, context_records)

        if call_llm:
            impact_assessment = self._call_llm(prompt)
            validation = self._validate_grounding(impact_assessment, context_records)
            affected_baselines_df, baseline_determination = self._determine_baselines(impact_assessment)
            impact_report_df = pd.DataFrame([{
                "artifact_id": x.get("artifact_id"),
                "artifact_type": x.get("artifact_type"),
                "impact_level": x.get("impact_level"),
                "candidate_category": x.get("candidate_category"),
                "traceability_status": x.get("traceability_status"),
                "confidence": x.get("confidence"),
                "reason": x.get("reason"),
            } for x in impact_assessment.get("assessments", [])])
        else:
            impact_assessment = None
            validation = None
            affected_baselines_df = pd.DataFrame()
            baseline_determination = None
            impact_report_df = pd.DataFrame()

        retrieval_summary_df = pd.DataFrame([{
            "artifact_type": t,
            "available": len(all_semantic_scores_by_type[t]),
            "semantic_retained": len(semantic_results_by_type[t]),
            "lexical_retained": len(lexical_results_by_type[t]),
        } for t in self.retrieval_types])

        return {
            "query": query,
            "impact_assessment": impact_assessment,
            "impact_report_df": impact_report_df,
            "overall_assessment": None if impact_assessment is None else impact_assessment.get("overall_assessment", ""),
            "affected_baselines_df": affected_baselines_df,
            "baseline_determination": baseline_determination,
            "validation": validation,
            "ranked_candidates_df": ranked,
            "selected_candidates_df": selected,
            "retrieval_summary_df": retrieval_summary_df,
            "prompt": prompt,
        }
