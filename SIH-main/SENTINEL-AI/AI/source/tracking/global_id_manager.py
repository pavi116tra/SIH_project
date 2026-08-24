"""
Global Identity Manager.
Implements Centralized Two-Level Identity Architecture: maps camera-local Track IDs (Level 1)
to persistent Global Person IDs P001, P002... (Level 2) across all cameras using a single
thread-safe Global Identity Gallery and ReID feature matching.
"""

import time
import threading
from datetime import datetime
import numpy as np
from reid.reid_extractor import PersonReIDExtractor


class GlobalPerson:
    """
    Represents a persistent Global Person identity (P001, P002...) across all cameras.
    """

    def __init__(self, global_id, initial_embedding, camera_id, local_track_id, source_type="file", timestamp=None, suspect_name=None):
        self.global_id = global_id  # e.g., "P001"
        self.suspect_name = suspect_name  # e.g., "PRAKALYA" or None
        self.first_seen = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_seen_ts = time.time()
        self.last_seen = self.first_seen

        self.embedding_history = []
        if initial_embedding is not None:
            self.embedding_history.append(initial_embedding)

        self.running_embedding = self._compute_running_embedding()

        # Structured tracks map per camera:
        # { "CAM01": {"track_id": 7, "confidence": 0.95, "last_seen_ts": ts, "source_type": "live"} }
        self.tracks = {}
        self.update_observation(initial_embedding, camera_id, local_track_id, source_type=source_type, sim_score=1.0)

    def _compute_running_embedding(self):
        if not self.embedding_history:
            return None
        mean_vec = np.mean(self.embedding_history, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            return mean_vec / norm
        return mean_vec

    def update_observation(self, embedding, camera_id, local_track_id, source_type="file", sim_score=1.0):
        """Update last seen time, add embedding sample, and update weighted running embedding."""
        now_ts = time.time()
        self.last_seen_ts = now_ts
        self.last_seen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update track reference for this camera
        self.tracks[camera_id] = {
            "track_id": local_track_id,
            "confidence": round(float(sim_score), 2),
            "last_seen_ts": now_ts,
            "source_type": source_type
        }

        if embedding is not None:
            self.embedding_history.append(embedding)
            if len(self.embedding_history) > 30:
                self.embedding_history.pop(0)

            # Weighted Moving Average for OSNet embeddings: 0.7 * old + 0.3 * new
            if self.running_embedding is None:
                self.running_embedding = embedding
            else:
                updated_vec = 0.7 * self.running_embedding + 0.3 * embedding
                norm = np.linalg.norm(updated_vec)
                if norm > 0:
                    self.running_embedding = updated_vec / norm
                else:
                    self.running_embedding = updated_vec

    def get_max_similarity(self, query_emb, reid_extractor):
        """Compute maximum cosine similarity against running embedding and stored history."""
        if query_emb is None:
            return 0.0

        max_sim = 0.0
        if self.running_embedding is not None:
            max_sim = reid_extractor.compute_similarity(query_emb, self.running_embedding)

        for ref_emb in self.embedding_history:
            sim = reid_extractor.compute_similarity(query_emb, ref_emb)
            if sim > max_sim:
                max_sim = sim
        return max_sim


class GlobalIDManager:
    """
    Manages Global Person identities across multi-camera streams using OSNet embeddings.
    Enforces a single central identity gallery, thread safety, periodic background merging,
    video-source loop handling, Suspect Ground Truth Anchoring, and Suspect Name Propagation.
    """

    def __init__(
        self,
        retention_minutes=60,
        reid_match_threshold=0.65,
        device="auto",
        auto_merge_interval=5.0
    ):
        self.lock = threading.Lock()
        self.retention_seconds = retention_minutes * 60
        self.reid_match_threshold = reid_match_threshold

        self.reid_extractor = PersonReIDExtractor(model_name="osnet_x1_0", device=device)

        self.global_people = {}  # {global_id_str: GlobalPerson}
        self.local_to_global_map = {}  # {(camera_id, local_track_id): global_id}
        self.suspect_to_global_map = {}  # {suspect_name: global_id_str}
        self.track_buffers = {}  # {(camera_id, local_track_id): [(timestamp, embedding), ...]}
        self.global_id_counter = 0

        # Start periodic background merge thread
        self.auto_merge_interval = auto_merge_interval
        self._running = True
        self._merge_thread = threading.Thread(target=self._background_merge_loop, daemon=True)
        self._merge_thread.start()

    def _generate_next_global_id(self):
        self.global_id_counter += 1
        return f"P{self.global_id_counter:03d}"

    def reset_camera_tracks(self, camera_id):
        """
        Hook called when a video source loops or is restarted.
        Resets track mappings for that camera while PRESERVING existing Global IDs.
        """
        with self.lock:
            keys_to_remove = [k for k in self.local_to_global_map if k[0] == camera_id]
            for k in keys_to_remove:
                del self.local_to_global_map[k]
                if k in self.track_buffers:
                    del self.track_buffers[k]
            print(f"[GlobalIDManager] Reset local track mappings for {camera_id} (Global IDs preserved).")

    def expire_camera_tracks(self, camera_id):
        """
        Hook called when a video file for camera_id is replaced/uploaded.
        Clears local track mappings for camera_id and purges any Global Person
        records that were exclusively tied to the replaced video's tracks.
        """
        with self.lock:
            keys_to_remove = [k for k in self.local_to_global_map if k[0] == camera_id]
            for k in keys_to_remove:
                del self.local_to_global_map[k]
                if k in self.track_buffers:
                    del self.track_buffers[k]

            gids_to_purge = []
            for gid, person in list(self.global_people.items()):
                if camera_id in person.tracks:
                    del person.tracks[camera_id]
                if not person.tracks and not person.suspect_name:
                    gids_to_purge.append(gid)

            for gid in gids_to_purge:
                del self.global_people[gid]

            print(f"[GlobalIDManager] Expired stale tracks and purged {len(gids_to_purge)} obsolete Global IDs for replaced {camera_id}.")

    def _get_representative_embedding(self, camera_id, local_track_id, emb, window_seconds=2.0):
        """Accumulates OSNet embeddings over a ~2-second window per track and returns L2-normalized mean."""
        now_ts = time.time()
        key = (camera_id, local_track_id)
        if key not in self.track_buffers:
            self.track_buffers[key] = []

        if emb is not None:
            self.track_buffers[key].append((now_ts, emb))

        # Keep only last 2 seconds of valid embeddings
        self.track_buffers[key] = [item for item in self.track_buffers[key] if (now_ts - item[0]) <= window_seconds and item[1] is not None]

        valid_embs = [item[1] for item in self.track_buffers[key]]
        if not valid_embs:
            return emb

        mean_emb = np.mean(valid_embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            return mean_emb / norm
        return mean_emb

    def merge_identities(self, source_gid, target_gid, similarity=0.0):
        """
        Merge source_gid into target_gid when OSNet ReID proves they represent the same person.
        Updates local_to_global_map, merges track references, and recalculates running embedding.
        """
        if source_gid == target_gid or source_gid not in self.global_people or target_gid not in self.global_people:
            return

        source_person = self.global_people.pop(source_gid)
        target_person = self.global_people[target_gid]

        # Preserve suspect_name anchor & propagate to target
        if source_person.suspect_name and not target_person.suspect_name:
            target_person.suspect_name = source_person.suspect_name
            if target_person.suspect_name:
                self.suspect_to_global_map[target_person.suspect_name] = target_gid

        # Transfer embeddings
        target_person.embedding_history.extend(source_person.embedding_history)
        if len(target_person.embedding_history) > 30:
            target_person.embedding_history = target_person.embedding_history[-30:]
        target_person.running_embedding = target_person._compute_running_embedding()

        # Transfer tracks
        for cam_id, track_info in source_person.tracks.items():
            if cam_id not in target_person.tracks or track_info["last_seen_ts"] > target_person.tracks[cam_id]["last_seen_ts"]:
                target_person.tracks[cam_id] = track_info

        # Update local_to_global_map
        for key, mapped_gid in list(self.local_to_global_map.items()):
            if mapped_gid == source_gid:
                self.local_to_global_map[key] = target_gid

        print(f"[GLOBAL ID MERGE] Merged duplicate identity {source_gid} into {target_gid} (OSNet Sim={similarity*100:.1f}%)")

    def run_merge_pass(self, merge_threshold=None):
        """
        Pairwise cosine similarity scan across all active Global IDs using OSNet embeddings.
        Merges redundant entries exceeding merge_threshold (defaults to self.reid_match_threshold).
        """
        if merge_threshold is None:
            merge_threshold = self.reid_match_threshold

        with self.lock:
            active_gids = list(self.global_people.keys())
            now_ts = time.time()
            merged_count = 0

            for i in range(len(active_gids)):
                gid1 = active_gids[i]
                if gid1 not in self.global_people:
                    continue

                person1 = self.global_people[gid1]
                if (now_ts - person1.last_seen_ts) > self.retention_seconds or person1.running_embedding is None:
                    continue

                for j in range(i + 1, len(active_gids)):
                    gid2 = active_gids[j]
                    if gid2 not in self.global_people:
                        continue

                    person2 = self.global_people[gid2]
                    if (now_ts - person2.last_seen_ts) > self.retention_seconds or person2.running_embedding is None:
                        continue

                    name1 = person1.suspect_name if (person1.suspect_name and person1.suspect_name != "Unknown") else None
                    name2 = person2.suspect_name if (person2.suspect_name and person2.suspect_name != "Unknown") else None

                    # HARD REJECTION: Never merge two DIFFERENT named suspects!
                    if name1 and name2 and name1 != name2:
                        print(f"[REID HARD BLOCK] appearance-similarity suggests match between {gid1} ('{name1}') and {gid2} ('{name2}') but suspect identities conflict — treating as different people")
                        continue

                    # AUTOMATIC SAME-SUSPECT MERGE: If both GIDs share the SAME named suspect, merge them instantly!
                    if name1 and name2 and name1 == name2:
                        target_g = min(gid1, gid2)
                        src_g = max(gid1, gid2)
                        self.merge_identities(src_g, target_g, similarity=1.0)
                        merged_count += 1
                        continue

                    sim = self.reid_extractor.compute_similarity(person1.running_embedding, person2.running_embedding)
                    if sim >= merge_threshold:
                        # If one is named, keep the named one as target
                        if name2 and not name1:
                            target_g = gid2
                            src_g = gid1
                        else:
                            target_g = min(gid1, gid2)
                            src_g = max(gid1, gid2)

                        self.merge_identities(src_g, target_g, similarity=sim)
                        merged_count += 1

            return merged_count

    def _background_merge_loop(self):
        """Background thread executing periodic merge passes every N seconds."""
        while self._running:
            time.sleep(self.auto_merge_interval)
            try:
                self.run_merge_pass(merge_threshold=self.reid_match_threshold)
            except Exception as e:
                print(f"[GlobalIDManager] Error in background merge loop: {e}")

    def get_or_assign_global_id(self, crop, camera_id, local_track_id, source_type="file", suspect_name=None, suspect_confidence=0.0):
        """
        Thread-safe method mapping a local track observation to a central Global Person ID using OSNet embeddings.
        If a confident suspect match exists (suspect_name != "Unknown"), suspect identity ANCHORS the Global ID.
        Also propagates suspect names across all tracks linked under the same Global ID.
        """
        with self.lock:
            key = (camera_id, local_track_id)
            emb = self.reid_extractor.extract_features(crop) if crop is not None else None
            rep_emb = self._get_representative_embedding(camera_id, local_track_id, emb)
            now_ts = time.time()

            # --- PATH A: Confident Suspect Match Ground Truth (Face Recognition Anchored) ---
            if suspect_name and suspect_name != "Unknown":
                prev_gid = self.local_to_global_map.get(key)

                if suspect_name in self.suspect_to_global_map:
                    target_gid = self.suspect_to_global_map[suspect_name]
                    print(f"[SUSPECT ANCHOR MATCH] Camera={camera_id} Track={local_track_id} Suspect='{suspect_name}' -> GlobalID={target_gid}")
                else:
                    # First time seeing this suspect anywhere: check if previous key had a temporary unknown ID
                    if prev_gid and prev_gid in self.global_people and not self.global_people[prev_gid].suspect_name:
                        target_gid = prev_gid
                    else:
                        target_gid = self._generate_next_global_id()

                    self.suspect_to_global_map[suspect_name] = target_gid
                    print(f"[SUSPECT ANCHOR NEW] Camera={camera_id} Track={local_track_id} Suspect='{suspect_name}' -> Created GlobalID={target_gid}")

                # If key was previously assigned to a DIFFERENT temporary Global ID, merge prev_gid into target_gid!
                if prev_gid and prev_gid != target_gid and prev_gid in self.global_people:
                    print(f"[SUSPECT ANCHOR CONSOLIDATION] Merging temporary {prev_gid} into suspect-anchored {target_gid} for '{suspect_name}'")
                    self.merge_identities(prev_gid, target_gid, similarity=1.0)

                # Ensure GlobalPerson record exists & is anchored
                if target_gid not in self.global_people:
                    init_emb = rep_emb if rep_emb is not None else emb
                    new_person = GlobalPerson(target_gid, init_emb, camera_id, local_track_id, source_type=source_type, suspect_name=suspect_name)
                    self.global_people[target_gid] = new_person
                else:
                    person = self.global_people[target_gid]
                    person.suspect_name = suspect_name
                    person.update_observation(rep_emb, camera_id, local_track_id, source_type=source_type, sim_score=1.0)

                self.local_to_global_map[key] = target_gid
                return target_gid

            # --- PATH B: Unrecognized / Unknown Person (Fallback to OSNet Appearance Re-ID) ---
            # 1. If key is already bound to a Global ID, update observation, check suspect propagation, and return
            if key in self.local_to_global_map:
                current_gid = self.local_to_global_map[key]
                if current_gid in self.global_people:
                    person = self.global_people[current_gid]
                    person.update_observation(rep_emb, camera_id, local_track_id, source_type=source_type, sim_score=1.0)

                    # SUSPECT NAME PROPAGATION: If Global ID acquired a suspect name on another camera, pass it back!
                    if person.suspect_name:
                        print(f"[SUSPECT PROPAGATED] {camera_id} / Track {local_track_id} -> Inherited Suspect '{person.suspect_name}' via Global ID {current_gid}")

                    return current_gid

            # 2. Search central Global Identity Gallery for eligible OSNet match
            best_gid = None
            best_sim = 0.0
            matrix_log = []

            for gid, person in self.global_people.items():
                if (now_ts - person.last_seen_ts) <= self.retention_seconds:
                    query_emb = rep_emb if rep_emb is not None else emb
                    if query_emb is not None:
                        sim = person.get_max_similarity(query_emb, self.reid_extractor)

                        # RULE 3a: Hard Negative Block - Conflicting confident suspect identities must NEVER match
                        if suspect_name and suspect_name != "Unknown" and person.suspect_name and person.suspect_name != "Unknown" and suspect_name != person.suspect_name:
                            print(f"[REID HARD BLOCK] appearance-similarity suggests match ({sim:.2f}) between candidate {gid} ('{person.suspect_name}') and track ('{suspect_name}') but suspect identities conflict — treating as different people")
                            continue

                        matrix_log.append(f"{gid}:{sim:.2f}")
                        if sim > best_sim:
                            best_sim = sim
                            best_gid = gid

            if matrix_log:
                print(f"[OSNet MATRIX] Camera={camera_id} Track={local_track_id} vs Gallery [{', '.join(matrix_log)}]")

            # 3. Match against existing Global ID if similarity >= threshold
            if best_gid is not None and best_sim >= self.reid_match_threshold:
                person = self.global_people[best_gid]
                old_tracks = list(person.tracks.items())
                person.update_observation(rep_emb, camera_id, local_track_id, source_type=source_type, sim_score=best_sim)
                self.local_to_global_map[key] = best_gid

                # RULE 3b: Flag unverified merges between named suspect and unknown track
                if (suspect_name and suspect_name != "Unknown" and (not person.suspect_name or person.suspect_name == "Unknown")) or \
                   ((not suspect_name or suspect_name == "Unknown") and person.suspect_name and person.suspect_name != "Unknown"):
                    print(f"[OSNet REID MATCH (UNVERIFIED)] Camera={camera_id} Track={local_track_id} ('{suspect_name or 'Unknown'}') -> Matched {best_gid} ('{person.suspect_name or 'Unknown'}') with Sim: {best_sim*100:.1f}%")
                else:
                    print(f"[OSNet REID MATCH]\n{camera_id} / Track {local_track_id} -> Matched {best_gid} (Sim: {best_sim*100:.1f}%)")

                if old_tracks and old_tracks[-1][0] != camera_id:
                    old_cam, old_info = old_tracks[-1]
                    print(f"[CAMERA TRANSITION]\n{best_gid}:\n{old_cam} / Track {old_info['track_id']} -> {camera_id} / Track {local_track_id}")

                return best_gid

            # 4. Create new Global Person ID if no match meets threshold
            new_gid = self._generate_next_global_id()
            init_emb = rep_emb if rep_emb is not None else emb
            new_person = GlobalPerson(new_gid, init_emb, camera_id, local_track_id, source_type=source_type)
            self.global_people[new_gid] = new_person
            self.local_to_global_map[key] = new_gid

            print(f"[OSNet REID NEW]\n{camera_id} / Track {local_track_id} -> Created {new_gid}")
            return new_gid

    def compute_and_print_pairwise_similarity_matrix(self):
        """
        Computes and prints the full pairwise cosine similarity matrix across all currently active
        Global Person records/tracks using OSNet embeddings. Labeled with camera + suspect name.
        """
        with self.lock:
            active_gids = [gid for gid, p in sorted(self.global_people.items()) if p.running_embedding is not None]
            if not active_gids:
                print("[OSNet PAIRWISE MATRIX] No active Global Person records with embeddings.")
                return None

            labels = []
            embeddings = []
            for gid in active_gids:
                p = self.global_people[gid]
                cam_info = ", ".join([f"{cam}:#{t['track_id']}" for cam, t in p.tracks.items()])
                s_name = p.suspect_name or "Unknown"
                labels.append(f"{gid} ({s_name}) [{cam_info}]")
                embeddings.append(p.running_embedding)

            n = len(embeddings)
            matrix = np.zeros((n, n), dtype=np.float32)

            for i in range(n):
                for j in range(n):
                    matrix[i, j] = self.reid_extractor.compute_similarity(embeddings[i], embeddings[j])

            print("\n" + "=" * 95)
            print("OSNet REAL-TIME PAIRWISE COSINE SIMILARITY MATRIX (ACTIVE GLOBAL IDENTITIES)")
            print("=" * 95)
            col_header = f"{'Global Person Identity':<35} | " + " | ".join([f"{labels[k][:12]:^12}" for k in range(n)])
            print(col_header)
            print("-" * len(col_header))

            for i in range(n):
                row_str = f"{labels[i]:<35} | "
                for j in range(n):
                    row_str += f"{matrix[i, j]:^12.4f} | "
                print(row_str)
            print("=" * 95 + "\n")

            return {"labels": labels, "matrix": matrix}

    def get_summary_analytics(self):
        """
        Get analytics dictionary of all active Global Person records in hierarchical nested format.
        """
        with self.lock:
            now_ts = time.time()
            records = []
            for gid, person in sorted(self.global_people.items()):
                is_active = (now_ts - person.last_seen_ts) <= self.retention_seconds
                if is_active:
                    camera_list = []
                    for cam_id, track_info in person.tracks.items():
                        camera_list.append({
                            "camera": cam_id,
                            "track_id": track_info["track_id"],
                            "confidence": track_info["confidence"],
                            "source_type": track_info.get("source_type", "file"),
                            "last_seen_ts": track_info.get("last_seen_ts", 0)
                        })

                    records.append({
                        "global_id": person.global_id,
                        "suspect_name": person.suspect_name or "Unknown",
                        "first_seen": person.first_seen,
                        "last_seen": person.last_seen,
                        "status": "ACTIVE",
                        "cameras": camera_list
                    })

            return {
                "total_unique_global_people": len(records),
                "records": records,
                "suspect_to_global_map": dict(self.suspect_to_global_map)
            }
