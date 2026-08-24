"""
Global Identity Manager.
Implements Centralized Two-Level Identity Architecture: maps camera-local Track IDs (Level 1)
to persistent Global Person IDs P001, P002... (Level 2) across all cameras using an Open-Set
Multi-Prototype Gallery, Two-Threshold Decision Logic (Accept/Reject/Pending), Temporal Aggregation,
New-ID Cooldown Immunity, and Real Similarity Score Tracking.
"""

import time
import threading
from datetime import datetime
import numpy as np
from reid.reid_extractor import PersonReIDExtractor


class GlobalPerson:
    """
    Represents a persistent Global Person identity (P001, P002...) across all cameras.
    Stores a multi-prototype gallery set of up to K=8 representative appearance feature vectors.
    """

    def __init__(self, global_id, initial_embedding, camera_id, local_track_id, source_type="file", timestamp=None, suspect_name=None, reid_extractor=None, max_prototypes=8):
        self.global_id = global_id  # e.g., "P001"
        self.suspect_name = suspect_name  # e.g., "PRAKALYA" or None
        self.first_seen = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = time.time()
        self.last_seen_ts = self.created_at
        self.last_seen = self.first_seen
        self.max_prototypes = max_prototypes

        # Multi-prototype gallery set (list of L2-normalized 512-dim numpy vectors)
        self.prototypes = []
        if initial_embedding is not None:
            self.prototypes.append(initial_embedding)

        # Structured tracks map per camera:
        # { "CAM01": {"track_id": 7, "confidence": 0.95, "last_seen_ts": ts, "source_type": "live"} }
        self.tracks = {}
        self.update_observation(initial_embedding, camera_id, local_track_id, source_type=source_type, sim_score=1.0, reid_extractor=reid_extractor)

    def add_prototype(self, embedding, reid_extractor=None, diversity_threshold=0.90):
        """
        Adds a candidate feature vector to the prototype gallery ONLY if it is sufficiently
        diverse (similarity to all existing prototypes < diversity_threshold=0.90).
        Drops the oldest prototype if prototype count exceeds self.max_prototypes.
        """
        if embedding is None:
            return False

        if not self.prototypes:
            self.prototypes.append(embedding)
            return True

        # Check diversity against existing prototypes
        if reid_extractor is not None:
            max_sim = max(reid_extractor.compute_similarity(embedding, proto) for proto in self.prototypes)
        else:
            max_sim = max(float(np.dot(embedding, proto)) for proto in self.prototypes)

        if max_sim < diversity_threshold:
            self.prototypes.append(embedding)
            if len(self.prototypes) > self.max_prototypes:
                self.prototypes.pop(0)  # Drop oldest prototype
            return True
        return False

    def update_observation(self, embedding, camera_id, local_track_id, source_type="file", sim_score=1.0, reid_extractor=None, diversity_threshold=0.90):
        """Update last seen time, add prototype if diverse, and store REAL similarity confidence."""
        now_ts = time.time()
        self.last_seen_ts = now_ts
        self.last_seen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update track reference for this camera (stores REAL similarity score, fixing 100% display bug)
        self.tracks[camera_id] = {
            "track_id": local_track_id,
            "confidence": round(float(sim_score), 2),
            "last_seen_ts": now_ts,
            "source_type": source_type
        }

        if embedding is not None:
            self.add_prototype(embedding, reid_extractor=reid_extractor, diversity_threshold=diversity_threshold)

    def get_max_similarity(self, query_emb, reid_extractor):
        """Compute maximum cosine similarity across all stored prototype feature vectors."""
        if query_emb is None or not self.prototypes:
            return 0.0

        max_sim = 0.0
        for proto in self.prototypes:
            sim = reid_extractor.compute_similarity(query_emb, proto)
            if sim > max_sim:
                max_sim = sim
        return max_sim


class GlobalIDManager:
    """
    Manages Global Person identities across multi-camera streams using OSNet embeddings.
    Implements Open-Set Multi-Prototype Re-ID Decision Logic:
      - ACCEPT_THRESHOLD (0.75): Confident match to existing Global ID.
      - REJECT_THRESHOLD (0.55): Confident novel person creation.
      - UNCERTAIN ZONE (0.55 < sim < 0.75): Mark track PENDING ("Identifying...") up to 5s max wait.
      - NEW-ID COOLDOWN (10s): Exclude fresh IDs from background merge passes.
      - STRICT MERGE THRESHOLD (0.85): Background merge pass safety limit.
    """

    def __init__(
        self,
        retention_minutes=60,
        accept_threshold=0.75,
        reject_threshold=0.55,
        merge_threshold=0.85,
        max_prototypes=8,
        cooldown_seconds=10.0,
        max_pending_wait_seconds=5.0,
        reid_match_threshold=None,
        device="auto",
        auto_merge_interval=5.0
    ):
        self.lock = threading.Lock()
        self.retention_seconds = retention_minutes * 60
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.merge_threshold = merge_threshold
        self.max_prototypes = max_prototypes
        self.cooldown_seconds = cooldown_seconds
        self.max_pending_wait_seconds = max_pending_wait_seconds

        self.reid_extractor = PersonReIDExtractor(model_name="osnet_x1_0", device=device)

        self.global_people = {}  # {global_id_str: GlobalPerson}
        self.local_to_global_map = {}  # {(camera_id, local_track_id): global_id}
        self.suspect_to_global_map = {}  # {suspect_name: global_id_str}
        self.track_buffers = {}  # {(camera_id, local_track_id): [(timestamp, embedding), ...]}
        self.pending_tracks = {}  # {(camera_id, local_track_id): first_seen_timestamp}
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
        """Hook called when a video source loops or is restarted."""
        with self.lock:
            keys_to_remove = [k for k in self.local_to_global_map if k[0] == camera_id]
            for k in keys_to_remove:
                del self.local_to_global_map[k]
                if k in self.track_buffers:
                    del self.track_buffers[k]
                if k in self.pending_tracks:
                    del self.pending_tracks[k]
            print(f"[GlobalIDManager] Reset local track mappings for {camera_id} (Global IDs preserved).")

    def expire_camera_tracks(self, camera_id):
        """Hook called when a video file for camera_id is replaced/uploaded."""
        with self.lock:
            keys_to_remove = [k for k in self.local_to_global_map if k[0] == camera_id]
            for k in keys_to_remove:
                del self.local_to_global_map[k]
                if k in self.track_buffers:
                    del self.track_buffers[k]
                if k in self.pending_tracks:
                    del self.pending_tracks[k]

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
            return emb, 0

        mean_emb = np.mean(valid_embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        rep_vec = mean_emb / norm if norm > 0 else mean_emb
        return rep_vec, len(valid_embs)

    def merge_identities(self, source_gid, target_gid, similarity=0.0):
        """
        Merge source_gid into target_gid when OSNet ReID proves they represent the same person.
        Consolidates prototypes, track history, and mapped keys.
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

        # Transfer prototypes (with diversity check)
        for proto in source_person.prototypes:
            target_person.add_prototype(proto, reid_extractor=self.reid_extractor)

        # Transfer tracks
        for cam_id, track_info in source_person.tracks.items():
            if cam_id not in target_person.tracks or track_info["last_seen_ts"] > target_person.tracks[cam_id]["last_seen_ts"]:
                target_person.tracks[cam_id] = track_info

        # Update local_to_global_map
        for key, mapped_gid in list(self.local_to_global_map.items()):
            if mapped_gid == source_gid:
                self.local_to_global_map[key] = target_gid

        print(f"[GLOBAL ID MERGE] Merged duplicate identity {source_gid} into {target_gid} (Sim={similarity*100:.1f}%)")

    def run_merge_pass(self, merge_threshold=None):
        """
        Pairwise multi-prototype similarity scan across all active Global IDs.
        Enforces:
          - New-ID Cooldown Immunity (skip IDs created in last 10 seconds).
          - Strict Merge Threshold (defaults to self.merge_threshold = 0.85).
          - Hard Negative Rejection (never merge conflicting named suspects).
        """
        if merge_threshold is None:
            merge_threshold = self.merge_threshold

        with self.lock:
            active_gids = list(self.global_people.keys())
            now_ts = time.time()
            merged_count = 0

            for i in range(len(active_gids)):
                gid1 = active_gids[i]
                if gid1 not in self.global_people:
                    continue

                person1 = self.global_people[gid1]

                # COOLDOWN CHECK: Exclude new Global IDs created within the last 10 seconds
                if (now_ts - person1.created_at) < self.cooldown_seconds:
                    continue

                if (now_ts - person1.last_seen_ts) > self.retention_seconds or not person1.prototypes:
                    continue

                for j in range(i + 1, len(active_gids)):
                    gid2 = active_gids[j]
                    if gid2 not in self.global_people:
                        continue

                    person2 = self.global_people[gid2]

                    # COOLDOWN CHECK: Exclude new Global IDs created within the last 10 seconds
                    if (now_ts - person2.created_at) < self.cooldown_seconds:
                        continue

                    if (now_ts - person2.last_seen_ts) > self.retention_seconds or not person2.prototypes:
                        continue

                    name1 = person1.suspect_name if (person1.suspect_name and person1.suspect_name != "Unknown") else None
                    name2 = person2.suspect_name if (person2.suspect_name and person2.suspect_name != "Unknown") else None

                    # HARD REJECTION: Never merge two DIFFERENT named suspects!
                    if name1 and name2 and name1 != name2:
                        continue

                    # AUTOMATIC SAME-SUSPECT MERGE: If both share the SAME named suspect, merge instantly!
                    if name1 and name2 and name1 == name2:
                        target_g = min(gid1, gid2)
                        src_g = max(gid1, gid2)
                        self.merge_identities(src_g, target_g, similarity=1.0)
                        merged_count += 1
                        continue

                    # Multi-prototype similarity check
                    sim = person1.get_max_similarity(person2.prototypes[0], self.reid_extractor)
                    for proto2 in person2.prototypes[1:]:
                        s = person1.get_max_similarity(proto2, self.reid_extractor)
                        if s > sim:
                            sim = s

                    if sim >= merge_threshold:
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
                self.run_merge_pass(merge_threshold=self.merge_threshold)
            except Exception as e:
                print(f"[GlobalIDManager] Error in background merge loop: {e}")

    def get_or_assign_global_id(self, crop, camera_id, local_track_id, source_type="file", suspect_name=None, suspect_confidence=0.0):
        """
        Thread-safe method mapping a local track observation to a central Global Person ID using Open-Set Decision Logic:
          - Path A (Suspect Match): Ground truth suspect face match anchors Global ID.
          - Path B (Open-Set ReID): Two-threshold decision (Accept >= 0.75, Reject <= 0.55, Pending 0.55..0.75).
        """
        with self.lock:
            key = (camera_id, local_track_id)
            emb = self.reid_extractor.extract_features(crop) if crop is not None else None
            rep_emb, num_samples = self._get_representative_embedding(camera_id, local_track_id, emb)
            now_ts = time.time()

            # --- PATH A: Confident Suspect Match Ground Truth (Face Recognition Anchored) ---
            if suspect_name and suspect_name != "Unknown":
                prev_gid = self.local_to_global_map.get(key)

                if suspect_name in self.suspect_to_global_map:
                    target_gid = self.suspect_to_global_map[suspect_name]
                else:
                    if prev_gid and prev_gid in self.global_people and not self.global_people[prev_gid].suspect_name:
                        target_gid = prev_gid
                    else:
                        target_gid = self._generate_next_global_id()

                    self.suspect_to_global_map[suspect_name] = target_gid
                    print(f"[SUSPECT ANCHOR NEW] Camera={camera_id} Track={local_track_id} Suspect='{suspect_name}' -> Created GlobalID={target_gid}")

                # Consolidate previous temporary Global ID if it differed from target_gid
                if prev_gid and prev_gid != target_gid and prev_gid in self.global_people:
                    print(f"[SUSPECT ANCHOR CONSOLIDATION] Merging temporary {prev_gid} into suspect-anchored {target_gid} for '{suspect_name}'")
                    self.merge_identities(prev_gid, target_gid, similarity=1.0)

                init_emb = rep_emb if rep_emb is not None else emb
                if target_gid not in self.global_people:
                    new_person = GlobalPerson(target_gid, init_emb, camera_id, local_track_id, source_type=source_type, suspect_name=suspect_name, reid_extractor=self.reid_extractor, max_prototypes=self.max_prototypes)
                    self.global_people[target_gid] = new_person
                else:
                    person = self.global_people[target_gid]
                    person.suspect_name = suspect_name
                    # Compute actual appearance similarity if available, else 0.95 (Face Ground Truth)
                    face_sim = person.get_max_similarity(init_emb, self.reid_extractor) if init_emb is not None else 0.95
                    person.update_observation(init_emb, camera_id, local_track_id, source_type=source_type, sim_score=max(face_sim, 0.90), reid_extractor=self.reid_extractor)

                self.local_to_global_map[key] = target_gid
                if key in self.pending_tracks:
                    del self.pending_tracks[key]
                return target_gid

            # --- PATH B: Open-Set Appearance Re-ID (Unregistered / Unknown Person) ---
            # 1. If key is ALREADY bound to a Global ID, update observation with real computed similarity and return
            if key in self.local_to_global_map:
                current_gid = self.local_to_global_map[key]
                if current_gid in self.global_people:
                    person = self.global_people[current_gid]
                    query_emb = rep_emb if rep_emb is not None else emb
                    real_sim = person.get_max_similarity(query_emb, self.reid_extractor) if query_emb is not None else 0.85
                    person.update_observation(query_emb, camera_id, local_track_id, source_type=source_type, sim_score=real_sim, reid_extractor=self.reid_extractor)
                    return current_gid

            # 2. TEMPORAL AGGREGATION BUFFER CHECK:
            # Require at least 2-3 buffered observation frames before executing open-set decision
            query_emb = rep_emb if rep_emb is not None else emb
            if query_emb is None and num_samples < 2:
                if key not in self.pending_tracks:
                    self.pending_tracks[key] = now_ts
                return "PENDING"

            # 3. Compute Max Prototype Similarity per active Global ID
            best_gid = None
            best_sim = 0.0
            matrix_log = []

            for gid, person in self.global_people.items():
                if (now_ts - person.last_seen_ts) <= self.retention_seconds:
                    if query_emb is not None:
                        sim = person.get_max_similarity(query_emb, self.reid_extractor)

                        # Hard Negative Block: Conflicting named suspects
                        if suspect_name and suspect_name != "Unknown" and person.suspect_name and person.suspect_name != "Unknown" and suspect_name != person.suspect_name:
                            continue

                        matrix_log.append(f"{gid}:{sim:.2f}")
                        if sim > best_sim:
                            best_sim = sim
                            best_gid = gid

            if matrix_log:
                print(f"[OSNet OPEN-SET MATRIX] Camera={camera_id} Track={local_track_id} vs Gallery [{', '.join(matrix_log)}]")

            # 4. TWO-THRESHOLD DECISION LOGIC WITH UNCERTAIN ZONE
            # CASE A: High Confidence Match (best_sim >= ACCEPT_THRESHOLD = 0.75)
            if best_gid is not None and best_sim >= self.accept_threshold:
                person = self.global_people[best_gid]
                person.update_observation(query_emb, camera_id, local_track_id, source_type=source_type, sim_score=best_sim, reid_extractor=self.reid_extractor)
                self.local_to_global_map[key] = best_gid
                if key in self.pending_tracks:
                    del self.pending_tracks[key]

                print(f"[OSNet OPEN-SET ACCEPT MATCH]\n{camera_id} / Track {local_track_id} -> Matched {best_gid} (Sim: {best_sim*100:.1f}%)")
                return best_gid

            # CASE B: High Confidence Novel Person Rejection (best_sim <= REJECT_THRESHOLD = 0.55)
            if best_sim <= self.reject_threshold or not self.global_people:
                new_gid = self._generate_next_global_id()
                new_person = GlobalPerson(new_gid, query_emb, camera_id, local_track_id, source_type=source_type, suspect_name=suspect_name, reid_extractor=self.reid_extractor, max_prototypes=self.max_prototypes)
                self.global_people[new_gid] = new_person
                self.local_to_global_map[key] = new_gid
                if key in self.pending_tracks:
                    del self.pending_tracks[key]

                print(f"[OSNet OPEN-SET REJECT NEW]\n{camera_id} / Track {local_track_id} -> Created New Global ID {new_gid} (Max Sim vs Gallery: {best_sim*100:.1f}%)")
                return new_gid

            # CASE C: UNCERTAIN ZONE (0.55 < best_sim < 0.75)
            if key not in self.pending_tracks:
                self.pending_tracks[key] = now_ts

            pending_duration = now_ts - self.pending_tracks[key]

            # Check if pending max wait timeout (5.0 seconds) reached
            if pending_duration < self.max_pending_wait_seconds:
                # Still within uncertain buffering window -> Keep track in PENDING state ("Identifying...")
                print(f"[OSNet UNCERTAIN ZONE PENDING] Camera={camera_id} Track={local_track_id} (Sim: {best_sim*100:.1f}%, Waiting {pending_duration:.1f}s / {self.max_pending_wait_seconds}s)")
                return "PENDING"
            else:
                # Max wait timeout reached! Default to creating a NEW Global ID (safer to split than to wrongly merge)
                new_gid = self._generate_next_global_id()
                new_person = GlobalPerson(new_gid, query_emb, camera_id, local_track_id, source_type=source_type, suspect_name=suspect_name, reid_extractor=self.reid_extractor, max_prototypes=self.max_prototypes)
                self.global_people[new_gid] = new_person
                self.local_to_global_map[key] = new_gid
                del self.pending_tracks[key]

                print(f"[OSNet UNCERTAIN TIMEOUT NEW]\n{camera_id} / Track {local_track_id} -> Created New Global ID {new_gid} after {pending_duration:.1f}s timeout (Sim: {best_sim*100:.1f}%)")
                return new_gid
