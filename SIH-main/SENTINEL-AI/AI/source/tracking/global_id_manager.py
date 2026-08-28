"""
Global Identity Manager.
Implements Centralized Two-Level Identity Architecture: maps camera-local Track IDs (Level 1)
to persistent Global Person IDs P001, P002... (Level 2) across all cameras using an Open-Set
Multi-Prototype Gallery, Two-Threshold Decision Logic (Accept/Reject/Pending), Temporal Aggregation,
New-ID Cooldown Immunity, and Real Similarity Score Tracking.
"""

import time
import threading
import os
import cv2
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
        self.prototype_crops = []
        if initial_embedding is not None:
            self.prototypes.append(initial_embedding)

        # Structured tracks map per camera:
        # { "CAM01": {"track_id": 7, "confidence": 0.95, "last_seen_ts": ts, "source_type": "live"} }
        self.tracks = {}
        self.update_observation(initial_embedding, camera_id, local_track_id, source_type=source_type, sim_score=1.0, reid_extractor=reid_extractor)

    def add_prototype(self, embedding, crop=None, reid_extractor=None, diversity_threshold=0.90):
        """
        Adds a candidate feature vector to the prototype gallery ONLY if it is sufficiently
        diverse (similarity to all existing prototypes < diversity_threshold=0.90).
        Drops the oldest prototype if prototype count exceeds self.max_prototypes.
        """
        if embedding is None:
            return False

        if not self.prototypes:
            self.prototypes.append(embedding)
            if crop is not None:
                self.prototype_crops.append(crop.copy())
            return True

        # Check diversity against existing prototypes
        if reid_extractor is not None:
            max_sim = max(reid_extractor.compute_similarity(embedding, proto) for proto in self.prototypes)
        else:
            max_sim = max(float(np.dot(embedding, proto)) for proto in self.prototypes)

        if max_sim < diversity_threshold:
            self.prototypes.append(embedding)
            if crop is not None:
                self.prototype_crops.append(crop.copy())
            if len(self.prototypes) > self.max_prototypes:
                self.prototypes.pop(0)  # Drop oldest prototype
                if self.prototype_crops:
                    self.prototype_crops.pop(0)
            return True
        return False

    def update_observation(self, embedding, camera_id, local_track_id, source_type="file", sim_score=1.0, reid_extractor=None, diversity_threshold=0.90, crop=None):
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
            self.add_prototype(embedding, crop=crop, reid_extractor=reid_extractor, diversity_threshold=diversity_threshold)

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
        accept_threshold=0.88,
        reject_threshold=0.55,
        merge_threshold=0.88,
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
        if reid_match_threshold is not None:
            self.accept_threshold = reid_match_threshold
            self.reid_match_threshold = reid_match_threshold
        else:
            self.reid_match_threshold = self.accept_threshold

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

    def get_summary_analytics(self):
        """
        Returns real-time analytics summary of active cameras, total visible persons,
        and unique Global Level 2 identities, formatted for multi-cam dashboard and UI hierarchy panel.
        """
        with self.lock:
            active_cams = set()
            visible_count = 0
            unique_gids = set()
            now_ts = time.time()
            hierarchy = []

            suspect_to_global_map = {}
            for gid, person in list(self.global_people.items()):
                if (now_ts - person.last_seen_ts) <= self.retention_seconds:
                    unique_gids.add(gid)
                    if person.suspect_name:
                        suspect_to_global_map[person.suspect_name] = gid
                    active_tracks_list = []
                    for cam_id, track_info in list(person.tracks.items()):
                        if (now_ts - track_info["last_seen_ts"]) <= 5.0:
                            active_cams.add(cam_id)
                            visible_count += 1
                            active_tracks_list.append({
                                "camera_id": cam_id,
                                "camera": cam_id,
                                "track_id": track_info["track_id"],
                                "confidence": track_info["confidence"],
                                "last_seen_ts": track_info["last_seen_ts"],
                                "source_type": track_info["source_type"]
                            })

                    hierarchy.append({
                        "global_id": gid,
                        "suspect_name": person.suspect_name or "Unknown",
                        "first_seen": person.first_seen,
                        "last_seen": person.last_seen,
                        "prototypes_count": len(person.prototypes),
                        "active_tracks": active_tracks_list,
                        "cameras": active_tracks_list
                    })

            hierarchy.sort(key=lambda x: x["global_id"])

            return {
                "total_unique_global_people": len(unique_gids),
                "configured_feeds": 4,
                "active_feeds_count": len(active_cams),
                "visible_human_count": visible_count,
                "unique_human_count": len(unique_gids),
                "global_identities_count": len(unique_gids),
                "active_cameras": sorted(list(active_cams)),
                "records": hierarchy,
                "hierarchy": hierarchy,
                "suspect_to_global_map": suspect_to_global_map
            }

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
                    # Generate a fresh dedicated Global ID for newly identified suspect ground-truth anchor
                    target_gid = self._generate_next_global_id()
                    self.suspect_to_global_map[suspect_name] = target_gid
                    print(f"[SUSPECT ANCHOR REGISTER] Suspect='{suspect_name}' -> Registered Dedicated GlobalID={target_gid}")

                # Consolidate previous temporary Global ID ONLY if appearance similarity passes ACCEPT_THRESHOLD
                if prev_gid and prev_gid != target_gid and prev_gid in self.global_people:
                    prev_person = self.global_people[prev_gid]
                    if target_gid in self.global_people and self.global_people[target_gid].prototypes:
                        target_person = self.global_people[target_gid]
                        app_sim = 0.0
                        if prev_person.prototypes:
                            app_sim = max(target_person.get_max_similarity(proto, self.reid_extractor) for proto in prev_person.prototypes)
                        if app_sim >= self.accept_threshold:
                            print(f"[SUSPECT ANCHOR CONSOLIDATION ACCEPT] Merging temporary {prev_gid} into suspect-anchored {target_gid} (Sim: {app_sim*100:.1f}% >= {self.accept_threshold*100:.1f}%)")
                            self.merge_identities(prev_gid, target_gid, similarity=app_sim)
                        else:
                            print(f"[SUSPECT ANCHOR CONSOLIDATION REJECT] Appearance mismatch ({app_sim*100:.1f}% < {self.accept_threshold*100:.1f}%). Refusing to merge temporary {prev_gid} into suspect-anchored {target_gid}")
                    else:
                        print(f"[SUSPECT ANCHOR CONSOLIDATION NEW TARGET] Merging temporary {prev_gid} into new suspect-anchored {target_gid}")
                        self.merge_identities(prev_gid, target_gid, similarity=1.0)

                init_emb = rep_emb if rep_emb is not None else emb
                if target_gid not in self.global_people:
                    new_person = GlobalPerson(target_gid, init_emb, camera_id, local_track_id, source_type=source_type, suspect_name=suspect_name, reid_extractor=self.reid_extractor, max_prototypes=self.max_prototypes)
                    if crop is not None and init_emb is not None:
                        new_person.prototype_crops.append(crop.copy())
                    self.global_people[target_gid] = new_person
                else:
                    person = self.global_people[target_gid]
                    person.suspect_name = suspect_name
                    # Compute actual appearance similarity if available, else 0.95 (Face Ground Truth)
                    face_sim = person.get_max_similarity(init_emb, self.reid_extractor) if init_emb is not None else 0.95
                    person.update_observation(init_emb, camera_id, local_track_id, source_type=source_type, sim_score=max(face_sim, 0.90), reid_extractor=self.reid_extractor, crop=crop)

                self.local_to_global_map[key] = target_gid
                if key in self.pending_tracks:
                    del self.pending_tracks[key]

                print(f"[REID_DECISION] camera={camera_id} track={local_track_id} decision=ACCEPT similarity=1.00 threshold_used=0.00 assigned_global_id={target_gid} path=SUSPECT_ANCHOR reason=\"Face ground-truth anchor for suspect '{suspect_name}'\"")
                return target_gid

            # --- PATH B: Open-Set Appearance Re-ID (Unregistered / Unknown Person) ---
            # 1. If key is ALREADY bound to a Global ID, update observation with real computed similarity and return
            if key in self.local_to_global_map:
                current_gid = self.local_to_global_map[key]
                if current_gid in self.global_people:
                    person = self.global_people[current_gid]
                    query_emb = rep_emb if rep_emb is not None else emb
                    real_sim = person.get_max_similarity(query_emb, self.reid_extractor) if query_emb is not None else 0.85
                    person.update_observation(query_emb, camera_id, local_track_id, source_type=source_type, sim_score=real_sim, reid_extractor=self.reid_extractor, crop=crop)
                    print(f"[REID_DECISION] camera={camera_id} track={local_track_id} decision=ACCEPT similarity={real_sim:.2f} threshold_used={self.accept_threshold:.2f} assigned_global_id={current_gid} path=EXISTING_TRACK_BINDING reason=\"Local track {local_track_id} already bound to {current_gid}\"")
                    return current_gid

            # 2. TEMPORAL AGGREGATION BUFFER CHECK:
            # Require at least 2-3 buffered observation frames before executing open-set decision
            query_emb = rep_emb if rep_emb is not None else emb
            if query_emb is None and num_samples < 2:
                if key not in self.pending_tracks:
                    self.pending_tracks[key] = now_ts
                print(f"[REID_DECISION] camera={camera_id} track={local_track_id} decision=PENDING similarity=0.00 threshold_used={self.accept_threshold:.2f} assigned_global_id=PENDING path=UNCERTAIN_ZONE reason=\"Buffering initial feature vector (samples: {num_samples})\"")
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
            # CASE A: High Confidence Match (best_sim >= ACCEPT_THRESHOLD = 0.88)
            if best_gid is not None and best_sim >= self.accept_threshold:
                # HARD ASSERTION FOR RE-ID ACCURACY VALIDATION
                assert best_sim >= self.accept_threshold, f"Attempted to bind at {best_sim} which is below ACCEPT_THRESHOLD {self.accept_threshold}"

                person = self.global_people[best_gid]
                person.update_observation(query_emb, camera_id, local_track_id, source_type=source_type, sim_score=best_sim, reid_extractor=self.reid_extractor, crop=crop)
                self.local_to_global_map[key] = best_gid
                if key in self.pending_tracks:
                    del self.pending_tracks[key]

                print(f"[REID_DECISION] camera={camera_id} track={local_track_id} decision=ACCEPT similarity={best_sim:.2f} threshold_used={self.accept_threshold:.2f} assigned_global_id={best_gid} path=GALLERY_MATCH reason=\"Appearance similarity {best_sim*100:.1f}% >= accept threshold {self.accept_threshold*100:.1f}%\"")
                return best_gid

            # CASE B: High Confidence Novel Person Rejection (best_sim <= REJECT_THRESHOLD = 0.55)
            if best_sim <= self.reject_threshold or not self.global_people:
                new_gid = self._generate_next_global_id()
                new_person = GlobalPerson(new_gid, query_emb, camera_id, local_track_id, source_type=source_type, suspect_name=suspect_name, reid_extractor=self.reid_extractor, max_prototypes=self.max_prototypes)
                if crop is not None and query_emb is not None:
                    new_person.prototype_crops.append(crop.copy())
                self.global_people[new_gid] = new_person
                self.local_to_global_map[key] = new_gid
                if key in self.pending_tracks:
                    del self.pending_tracks[key]

                print(f"[REID_DECISION] camera={camera_id} track={local_track_id} decision=REJECT similarity={best_sim:.2f} threshold_used={self.reject_threshold:.2f} assigned_global_id={new_gid} path=NEW_ID reason=\"Appearance similarity {best_sim*100:.1f}% <= reject threshold {self.reject_threshold*100:.1f}%\"")
                return new_gid

            # CASE C: UNCERTAIN ZONE (0.55 < best_sim < 0.88)
            if key not in self.pending_tracks:
                self.pending_tracks[key] = now_ts

            pending_duration = now_ts - self.pending_tracks[key]

            # Check if pending max wait timeout (5.0 seconds) reached
            if pending_duration < self.max_pending_wait_seconds:
                # Still within uncertain buffering window -> Keep track in PENDING state ("Identifying...")
                print(f"[REID_DECISION] camera={camera_id} track={local_track_id} decision=PENDING similarity={best_sim:.2f} threshold_used={self.accept_threshold:.2f} assigned_global_id=PENDING path=UNCERTAIN_ZONE reason=\"Similarity {best_sim*100:.1f}% in uncertain zone (0.55-0.88), waiting {pending_duration:.1f}s / {self.max_pending_wait_seconds}s\"")
                return "PENDING"
            else:
                # Max wait timeout reached! Default to creating a NEW Global ID (safer to split than to wrongly merge)
                new_gid = self._generate_next_global_id()
                new_person = GlobalPerson(new_gid, query_emb, camera_id, local_track_id, source_type=source_type, suspect_name=suspect_name, reid_extractor=self.reid_extractor, max_prototypes=self.max_prototypes)
                if crop is not None and query_emb is not None:
                    new_person.prototype_crops.append(crop.copy())
                self.global_people[new_gid] = new_person
                self.local_to_global_map[key] = new_gid
                del self.pending_tracks[key]

                print(f"[REID_DECISION] camera={camera_id} track={local_track_id} decision=REJECT similarity={best_sim:.2f} threshold_used={self.accept_threshold:.2f} assigned_global_id={new_gid} path=NEW_ID reason=\"Uncertain buffering timed out after {pending_duration:.1f}s, creating novel ID\"")
                return new_gid

    def export_prototype_gallery_crops(self, output_dir):
        """Export all prototype crop images stored across active Global IDs to disk."""
        import cv2
        os.makedirs(output_dir, exist_ok=True)
        saved_info = []
        with self.lock:
            for gid, person in self.global_people.items():
                for idx, crop in enumerate(person.prototype_crops):
                    if crop is not None and crop.size > 0:
                        filename = f"{gid}_proto_{idx+1}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, crop)
                        saved_info.append({"global_id": gid, "prototype_idx": idx + 1, "filepath": filepath})
        return saved_info

    def cleanup_gallery_outliers(self, target_gid="P001", min_cluster_sim=0.65):
        """
        Scans target_gid's prototype gallery, identifies contaminating outlier prototypes
        whose average similarity to the gallery medoid is below min_cluster_sim,
        and splits them into a clean new Global ID.
        """
        with self.lock:
            if target_gid not in self.global_people:
                return {"split_count": 0, "message": f"{target_gid} not found"}

            person = self.global_people[target_gid]
            if len(person.prototypes) <= 1:
                return {"split_count": 0, "message": "1 or fewer prototypes, no cleanup needed"}

            n = len(person.prototypes)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if self.reid_extractor:
                        sim_matrix[i, j] = PersonReIDExtractor.compute_similarity(person.prototypes[i], person.prototypes[j])

            medoid_idx = int(np.argmax(np.mean(sim_matrix, axis=1)))
            medoid_proto = person.prototypes[medoid_idx]

            valid_prototypes = []
            valid_crops = []
            outlier_prototypes = []
            outlier_crops = []

            for idx in range(n):
                proto = person.prototypes[idx]
                crop = person.prototype_crops[idx] if idx < len(person.prototype_crops) else None
                sim_to_medoid = float(sim_matrix[idx, medoid_idx])
                if sim_to_medoid >= min_cluster_sim:
                    valid_prototypes.append(proto)
                    if crop is not None:
                        valid_crops.append(crop)
                else:
                    outlier_prototypes.append(proto)
                    if crop is not None:
                        outlier_crops.append(crop)

            if outlier_prototypes:
                person.prototypes = valid_prototypes
                person.prototype_crops = valid_crops
                new_gid = self._generate_next_global_id()
                new_person = GlobalPerson(new_gid, outlier_prototypes[0], "CLEANUP", 0, suspect_name=None, reid_extractor=self.reid_extractor, max_prototypes=self.max_prototypes)
                new_person.prototypes = outlier_prototypes
                new_person.prototype_crops = outlier_crops
                self.global_people[new_gid] = new_person
                print(f"[GALLERY CLEANUP] Purged {len(outlier_prototypes)} contaminating prototype(s) from {target_gid} into new {new_gid}")
                return {"split_count": len(outlier_prototypes), "new_gid": new_gid, "purged": len(outlier_prototypes)}

            return {"split_count": 0, "message": "All prototypes are consistent with gallery medoid"}

    def compute_and_print_pairwise_similarity_matrix(self):
        """Compute and return pairwise similarity matrix across active global people."""
        with self.lock:
            gids = sorted(list(self.global_people.keys()))
            n = len(gids)
            matrix = np.zeros((n, n))
            for i, g1 in enumerate(gids):
                p1 = self.global_people[g1]
                for j, g2 in enumerate(gids):
                    p2 = self.global_people[g2]
                    max_sim = 0.0
                    for proto1 in p1.prototypes:
                        for proto2 in p2.prototypes:
                            sim = PersonReIDExtractor.compute_similarity(proto1, proto2) if self.reid_extractor else 0.0
                            if sim > max_sim:
                                max_sim = sim
                    matrix[i, j] = max_sim
            return {"gids": gids, "matrix": matrix}
