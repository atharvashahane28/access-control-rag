"""Small adversarial cases for permission and reference semantics, without downloads."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.organization import (build_workload, file_hash, load_organization,
                              make_roles_users, validate_organization)
from src.reference import exact_permitted_search, relevance_metrics


class PermissionTests(unittest.TestCase):
    def test_seeded_roles_and_multi_role_union(self):
        settings = {"n_departments": 6,
                    "departments_per_role": {"low": 1, "medium": 2, "high": 4},
                    "users_per_home": {"low": 1, "medium": 2, "high": 2}}
        roles, users = make_roles_users(settings, 42)
        self.assertEqual((roles, users), make_roles_users(settings, 42))
        self.assertNotEqual(roles, make_roles_users(settings, 43)[0])
        role_map = {r["role_id"]: set(r["department_ids"]) for r in roles}
        for user in users:
            expected = set().union(*(role_map[r] for r in user["role_ids"]))
            self.assertEqual(expected, set(user["department_ids"]))
            self.assertIn(user["home_department"], expected)
            if len(user["role_ids"]) > 1:
                self.assertGreater(len(expected), len(role_map[user["role_ids"][0]]))
        self.assertTrue(any(len(u["role_ids"]) > 1 for u in users))

    def test_eligibility_and_mixed_labels_use_only_permitted_evidence(self):
        ids = ["p0", "p1", "p2", "p3"]
        departments = np.array([0, 1, 2, 0])
        qids = ["home", "cross", "mixed", "denied"]
        qrels = pd.DataFrame({"question_id": ["home", "cross", "mixed", "mixed", "denied"],
                              "passage_id": ["p0", "p1", "p0", "p1", "p2"]})
        roles = [{"role_id": "low", "department_ids": [0]},
                 {"role_id": "medium", "department_ids": [0, 1]},
                 {"role_id": "high", "department_ids": [0, 1, 2]}]
        users = [{"user_id": level, "access_level": level, "home_department": 0,
                  "department_ids": role["department_ids"], "role_ids": [level]}
                 for level, role in zip(("low", "medium", "high"), roles)]
        workload = build_workload(ids, qids, departments, qrels, users)
        validate_organization(departments, roles, users, workload, len(ids), 3)
        low = workload[workload.user_id == "low"].set_index("question_id")
        self.assertEqual(set(low.index), {"home", "mixed"})
        self.assertEqual(low.loc["mixed", "question_type"], "in_department")
        self.assertEqual(low.loc["mixed", "n_relevant_permitted"], 1)
        medium = workload[workload.user_id == "medium"].set_index("question_id")
        self.assertEqual(medium.loc["cross", "question_type"], "cross_department")
        self.assertEqual(medium.loc["mixed", "question_type"], "mixed")
        self.assertNotIn("denied", medium.index)
        self.assertEqual(set(workload.question_id), set(qids))
        bad_users = [dict(users[0], department_ids=[0, 1]), *users[1:]]
        with self.assertRaisesRegex(ValueError, "union"):
            validate_organization(departments, roles, bad_users, workload, len(ids), 3)

    def test_missing_question_coverage_is_rejected(self):
        qrels = pd.DataFrame({"question_id": ["q"], "passage_id": ["p"]})
        with self.assertRaisesRegex(ValueError, "cover every"):
            build_workload(["p"], ["q"], np.array([1]), qrels,
                           [{"user_id": "u", "access_level": "low", "home_department": 0,
                             "department_ids": [0]}])

    def test_saved_workload_tampering_and_stale_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            files = {"passage_departments.csv": "passage_id,department_id\np,0\n",
                     "roles.json": "[]", "users.json": "[]",
                     "workload.csv": "workload_id,question_id,user_id\n0,q,u\n"}
            for name, content in files.items():
                (folder / name).write_text(content, encoding="utf-8")
            signature = {"seed": 42}
            (folder / "meta.json").write_text(json.dumps({"inputs": signature,
                "artifact_hashes": {name: file_hash(folder / name) for name in files}}), encoding="utf-8")
            with patch("src.organization.organization_dir", return_value=folder), patch(
                    "src.organization.input_signature", return_value=signature):
                self.assertEqual(len(load_organization({})["workload"]), 1)
                (folder / "workload.csv").write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    load_organization({})
            with patch("src.organization.organization_dir", return_value=folder), patch(
                    "src.organization.input_signature", return_value={"seed": 43}):
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    load_organization({})


class ExactReferenceTests(unittest.TestCase):
    def test_filtered_search_matches_independent_brute_force_and_pads(self):
        passages = np.array([[1, 0], [.8, .6], [0, 1], [-1, 0]], dtype=np.float32)
        queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
        allowed = np.array([1, 2, 3])  # nearest unrestricted passage is forbidden
        rows, scores, _, _ = exact_permitted_search(passages, queries, allowed, 5)
        expected_scores = queries @ passages[allowed].T
        order = np.argsort(-expected_scores, axis=1, kind="stable")
        np.testing.assert_array_equal(rows[:, :3], allowed[order])
        np.testing.assert_allclose(scores[:, :3], np.take_along_axis(expected_scores, order, axis=1))
        self.assertTrue(np.all(rows[:, 3:] == -1))
        self.assertTrue(np.all(np.isneginf(scores[:, 3:])))
        self.assertNotIn(0, rows)

    def test_empty_and_duplicate_permissions_rejected(self):
        vectors = np.eye(2, dtype=np.float32)
        for allowed in ([], [0, 0], [-1], [2]):
            with self.assertRaises(ValueError):
                exact_permitted_search(vectors, vectors[:1], allowed, 2)

    def test_relevance_metrics_are_not_neighbor_overlap(self):
        result = relevance_metrics(["irrelevant", "relevant_a"], {"relevant_a", "relevant_b"}, 2)
        self.assertEqual(result["relevant_hit"], 1)
        self.assertEqual(result["relevance_recall"], .5)
        self.assertAlmostEqual(result["ndcg"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertEqual(relevance_metrics([], {"relevant_a"}, 2)["relevance_recall"], 0)


if __name__ == "__main__":
    unittest.main()
