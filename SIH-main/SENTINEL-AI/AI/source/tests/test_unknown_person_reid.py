"""
Unit Test for Unknown / Unregistered Person Cross-Camera Re-ID Association.
Verifies that unregistered 3rd and 4th persons correctly match their Global IDs
across different camera streams (CAM02, CAM03, CAM04) while maintaining strict isolation
from registered suspects and other unknown individuals.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracking.global_id_manager import GlobalIDManager


def generate_normalized_embedding(seed=42):
    np.random.seed(seed)
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


def generate_similar_embedding(base_vec, similarity_target=0.78):
    """Generates an embedding vector with a controlled cosine similarity to base_vec."""
    np.random.seed(int(base_vec[0] * 1000) % 10000)
    noise = np.random.randn(512).astype(np.float32)
    # Orthogonalize noise with respect to base_vec
    noise = noise - np.dot(noise, base_vec) * base_vec
    noise = noise / np.linalg.norm(noise)

    # Combine: sim * base + sqrt(1 - sim^2) * noise
    sim = similarity_target
    combined = sim * base_vec + np.sqrt(1.0 - sim**2) * noise
    return (combined / np.linalg.norm(combined)).astype(np.float32)


def test_unknown_person_cross_cam_reid():
    print("=== STARTING UNKNOWN PERSON CROSS-CAMERA RE-ID TEST ===")

    # Initialize GlobalIDManager with optimal unknown person ReID threshold (0.65)
    manager = GlobalIDManager(reid_match_threshold=0.65, device="cpu")

    # Base embeddings for 4 individuals
    emb_pavitra_cam2 = generate_normalized_embedding(seed=101)
    emb_pavitra_cam4 = generate_similar_embedding(emb_pavitra_cam2, similarity_target=0.85)

    emb_prakalya_cam1 = generate_normalized_embedding(seed=202)
    emb_prakalya_cam3 = generate_similar_embedding(emb_prakalya_cam1, similarity_target=0.85)

    # Unknown Person 3 (Unregistered, appears on CAM02 and CAM03)
    emb_unk3_cam2 = generate_normalized_embedding(seed=303)
    emb_unk3_cam3 = generate_similar_embedding(emb_unk3_cam2, similarity_target=0.76) # 0.76 sim across cameras

    # Unknown Person 4 (Different Unregistered Person, appears on CAM03)
    emb_unk4_cam3 = generate_normalized_embedding(seed=404) # Distinct embedding (sim ~ 0.40)

    # 1. Register Suspect 1 (Pavitra S) on CAM04 -> Should get P001
    gid_p1_cam4 = manager.get_or_assign_global_id(
        crop=None, camera_id="CAM04", local_track_id=1, suspect_name="Pavitra S"
    )
    print(f"1. Pavitra S on CAM04  -> {gid_p1_cam4}")
    assert gid_p1_cam4 == "P001"

    # 2. Register Suspect 2 (Prakalya) on CAM01 -> Should get P002
    gid_p2_cam1 = manager.get_or_assign_global_id(
        crop=None, camera_id="CAM01", local_track_id=1, suspect_name="Prakalya"
    )
    print(f"2. Prakalya on CAM01   -> {gid_p2_cam1}")
    assert gid_p2_cam1 == "P002"

    # 3. Unknown Person 3 appears on CAM02 (First time seen) -> Should get P003
    # Inject representative embedding
    now_ts = time.time()
    manager.track_buffers[("CAM02", 10)] = [(now_ts, emb_unk3_cam2)]
    gid_unk3_cam2 = manager.get_or_assign_global_id(
        crop=None, camera_id="CAM02", local_track_id=10, suspect_name="Unknown"
    )
    print(f"3. Unknown #3 on CAM02 -> {gid_unk3_cam2}")
    assert gid_unk3_cam2 == "P003"

    # 4. Unknown Person 3 moves to CAM03 (Cross-Camera Match!) -> Should MATCH P003!
    now_ts = time.time()
    manager.track_buffers[("CAM03", 22)] = [(now_ts, emb_unk3_cam3)]
    sim_test = manager.reid_extractor.compute_similarity(emb_unk3_cam2, emb_unk3_cam3)
    print(f"Debug: Similarity between Unknown #3 CAM02 & CAM03 embeddings = {sim_test:.4f} (Threshold={manager.reid_match_threshold})")
    gid_unk3_cam3 = manager.get_or_assign_global_id(
        crop=None, camera_id="CAM03", local_track_id=22, suspect_name="Unknown"
    )
    print(f"4. Unknown #3 on CAM03 -> {gid_unk3_cam3} (Cross-Cam Match)")
    assert gid_unk3_cam3 == "P003", f"Expected P003 cross-camera match, but got {gid_unk3_cam3}!"

    # 5. Unknown Person 4 appears on CAM03 (Different Person) -> Should get P004
    now_ts = time.time()
    manager.track_buffers[("CAM03", 33)] = [(now_ts, emb_unk4_cam3)]
    gid_unk4_cam3 = manager.get_or_assign_global_id(
        crop=None, camera_id="CAM03", local_track_id=33, suspect_name="Unknown"
    )
    print(f"5. Unknown #4 on CAM03 -> {gid_unk4_cam3}")
    assert gid_unk4_cam3 == "P004", f"Expected P004, but got {gid_unk4_cam3}!"

    print("\n=== ALL UNKNOWN PERSON CROSS-CAMERA RE-ID TESTS PASSED 100%! ===")


if __name__ == "__main__":
    test_unknown_person_cross_cam_reid()
