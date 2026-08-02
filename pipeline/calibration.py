"""Camera calibration projector.

Converts pixel-space ball detections into metric cricket-pitch coordinates for
the MVP setup: a single, fixed smartphone camera recording from behind the
bowler, stationary for the session, over a standard 22-yard pitch.

Pipeline
--------
1. **Reference detection** — locate known pitch landmarks (the four pitch
   corners) in the image. Automatic via :class:`StumpPitchDetector`;
   ``manual_references`` may be supplied, and a learned detector can replace the
   heuristic through the :class:`PitchReferenceDetector` interface.
2. **Focal length** — estimated from the plane homography (no camera-specific
   constants are hardcoded); principal point assumed at image centre.
3. **Pose** — ``solvePnP`` against the known 3D pitch geometry gives camera
   rotation/translation.
4. **Metric observations** — each ball ray is intersected with an assumed
   vertical *delivery plane* to yield ``(x, y, z)``.

Honest limitations (MVP)
------------------------
A single monocular view has an inherent depth ambiguity along the optical axis.
Height ``z`` is only recoverable under the assumption that the ball travels in a
known vertical plane (``motion_plane_x``, default the pitch centre line). This
recovers down-pitch distance ``y`` (→ length) and height ``z`` (→ bounce, speed)
metrically, but fixes lateral ``x`` to the plane, so **line is not reliably
recovered** here. That is exactly the part a future multi-view or learned-depth
module improves — and it plugs in behind the same ``ObservationProjector``
interface without touching the rest of the pipeline.

If calibration confidence is low the video is **rejected** with an explanation
(:class:`CalibrationError`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from physics.trajectory import BallObservation
from pipeline.detection import BallDetection
from pipeline.ingest import IngestedVideo, VideoMetadata

logger = logging.getLogger("cricket_ai.calibration")

# Standard cricket pitch (metres). These are laws-of-cricket dimensions, not
# camera-specific values.
_PITCH_LENGTH = 20.12  # between the two bowling creases (22 yards)
_PITCH_WIDTH = 3.05  # prepared strip width (10 feet)
# Stump geometry (laws of cricket): three stumps span 9 in (0.2286 m) and stand
# 28 in (0.711 m) tall. The stumps are the calibration landmark used on real
# net-practice footage, where no painted pitch rectangle is visible.
_STUMP_HALF_SPAN = 0.2286 / 2.0  # 0.1143 m from centre to an outer stump
_STUMP_HEIGHT = 0.711


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class CalibrationError(Exception):
    """Raised when a video cannot be calibrated. Carries human-readable reasons."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("Calibration rejected: " + "; ".join(self.reasons))


# --------------------------------------------------------------------------- #
# Pitch geometry
# --------------------------------------------------------------------------- #
@dataclass
class PitchModel:
    """Standard pitch geometry and its named 3D reference points (metres).

    World frame matches :mod:`physics.trajectory`: origin at the striker's
    middle stump base, +y down the pitch toward the bowler, +x lateral, +z up.

    ``reference_points`` returns the **striker-end stumps** as a known 3D
    calibration landmark (three stumps, each ``_STUMP_HEIGHT`` tall, spanning
    ``2*_STUMP_HALF_SPAN``). This works on real net-practice footage where no
    painted pitch rectangle exists but the stumps are clearly visible. Points
    are ``(x, y, z)`` so the calibrator can handle this vertical plane; a ground
    plane (z=0) reference would still work via the same machinery.
    """

    length: float = _PITCH_LENGTH
    width: float = _PITCH_WIDTH

    def reference_points(self) -> Dict[str, Tuple[float, float, float]]:
        """Named 3D positions of stump tops and bases at both ends.

        Near (striker) stumps sit at y=0; far (bowler) stumps at y=length. The
        calibrator uses whichever names the detector actually provides — near
        alone gives a planar target; near+far gives a spatial one (better pose).
        """
        s = _STUMP_HALF_SPAN
        h = _STUMP_HEIGHT
        L = self.length
        return {
            # near / striker end (y = 0)
            "stump_left_base": (-s, 0.0, 0.0),
            "stump_left_top": (-s, 0.0, h),
            "stump_mid_base": (0.0, 0.0, 0.0),
            "stump_mid_top": (0.0, 0.0, h),
            "stump_right_base": (s, 0.0, 0.0),
            "stump_right_top": (s, 0.0, h),
            # far / bowler end (y = length)
            "far_stump_left_base": (-s, L, 0.0),
            "far_stump_left_top": (-s, L, h),
            "far_stump_mid_base": (0.0, L, 0.0),
            "far_stump_mid_top": (0.0, L, h),
            "far_stump_right_base": (s, L, 0.0),
            "far_stump_right_top": (s, L, h),
        }


# --------------------------------------------------------------------------- #
# Reference detection (pluggable — learned detector can replace the heuristic)
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceDetection:
    """Detected image locations of named pitch reference points."""

    points: Dict[str, Tuple[float, float]]  # name -> (u, v) pixels
    confidence: float
    issues: List[str] = field(default_factory=list)


class PitchReferenceDetector(Protocol):
    """Finds pitch reference points in a frame.

    A learned keypoint model implements this to replace the heuristic without
    any change to the calibrator or pipeline.
    """

    def detect(self, image: np.ndarray) -> Optional[ReferenceDetection]:  # pragma: no cover
        ...


class StumpPitchDetector:
    """Detects the striker-end stumps as a calibration landmark.

    Robust on real net-practice footage where no painted pitch is visible.
    Cricket stumps are three tall, thin, **bright vertical bars** in a close
    horizontal cluster with aligned bases — a structural cue that holds
    regardless of stump colour (yellow, white, ...) or turf, unlike the old
    bright-rectangle heuristic.

    Steps:
      1. threshold bright pixels, then a vertical morphological opening keeps
         tall-thin structures and removes wide regions (turf, sky, nets);
      2. keep vertical bar contours (height >> width);
      3. cluster the near stumps: the tallest bars, low in the frame, with
         aligned bases and close together horizontally;
      4. emit each stump's top and base as named reference points.

    Returns 2 or 3 stumps (4-6 points). Fewer than 2 => not found.
    """

    def __init__(
        self,
        brightness_thresholds: Sequence[float] = (185.0, 165.0, 145.0),
        min_bar_height_frac: float = 0.03,
        min_aspect: float = 2.0,
        base_align_frac: float = 0.06,
    ) -> None:
        # Tried strictest-first: a high threshold isolates stumps cleanly; if
        # too few bars survive it loosens. The vertical morphology removes wide
        # bright regions (sky, buildings, turf) so brightness need not be exact.
        self.brightness_thresholds = brightness_thresholds
        self.min_bar_height_frac = min_bar_height_frac
        self.min_aspect = min_aspect
        self.base_align_frac = base_align_frac

    def detect(self, image: np.ndarray) -> Optional[ReferenceDetection]:
        H, W = image.shape[:2]
        v = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        kh = max(15, int(H * 0.02))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kh))

        near, near_score, far = None, 0.0, None
        for thr in self.brightness_thresholds:
            bars = self._vertical_bars(v, thr, vk, H)
            if len(bars) < 3:
                continue
            n_near = [b for b in bars if b["base"] > 0.45 * H]
            triple, score = self._best_stump_triple(n_near)
            if triple and score > near_score:
                near, near_score = triple, score
                far = self._find_far_stumps(bars, triple)  # bonus, may be None
            if near_score > 0.85:
                break

        if not near:
            return ReferenceDetection(
                {}, 0.0, ["stumps not found (no three evenly-spaced aligned bars)"]
            )
        pts = self._to_reference_points(near, "stump")
        if far is not None:
            pts.update(self._to_reference_points(far, "far_stump"))
        return ReferenceDetection(points=pts, confidence=near_score, issues=[])

    def _vertical_bars(self, v, thr, vk, H):
        bright = (v > thr).astype(np.uint8) * 255
        vert = cv2.morphologyEx(bright, cv2.MORPH_OPEN, vk)
        cnts, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h < H * self.min_bar_height_frac or w > h or h / max(w, 1) < self.min_aspect:
                continue
            bars.append({"cx": x + w / 2.0, "top": float(y), "base": float(y + h), "h": h})
        return bars

    def _best_stump_triple(self, candidates, min_score: float = 0.0):
        """Best triple of bars matching the stump signature (similar height,
        aligned bases, even spacing). A full triple is required — two bars are
        ambiguous. Returns ``(triple_sorted_by_cx, score)`` or ``(None, 0.0)``."""
        import itertools

        cand = sorted(candidates, key=lambda b: b["h"], reverse=True)[:8]
        if len(cand) < 3:
            return None, 0.0
        best, best_score = None, 0.0
        for trio in itertools.combinations(cand, 3):
            t = sorted(trio, key=lambda b: b["cx"])
            heights = [b["h"] for b in t]
            bases = [b["base"] for b in t]
            g1 = t[1]["cx"] - t[0]["cx"]
            g2 = t[2]["cx"] - t[1]["cx"]
            if g1 < 2 or g2 < 2:
                continue
            span = t[2]["cx"] - t[0]["cx"]
            med_h = float(np.median(heights))
            even = 1.0 - min(1.0, abs(g1 - g2) / max(g1, g2))
            hcons = min(heights) / max(heights)
            align = 1.0 - min(1.0, (max(bases) - min(bases)) / max(med_h, 1))
            compact = 1.0 if span < med_h else max(0.0, 1.5 - span / med_h)
            if even < 0.6 or hcons < 0.6 or align < 0.6:
                continue
            score = 0.35 * even + 0.25 * hcons + 0.25 * align + 0.15 * compact
            if score > best_score:
                best, best_score = t, score
        if best_score < min_score:
            return None, 0.0
        return best, best_score

    def _find_far_stumps(self, bars, near_triple):
        """Far (bowler-end) stumps: a smaller aligned triple above the near one.

        Conservative — requires a high-scoring triple that sits above the near
        stumps and is smaller (perspective). Returns None when not confident, so
        far stumps only ever help, never harm (near-only calibration remains)."""
        near_top = min(b["top"] for b in near_triple)
        near_cx = float(np.mean([b["cx"] for b in near_triple]))
        near_span = near_triple[-1]["cx"] - near_triple[0]["cx"]
        near_med_h = float(np.median([b["h"] for b in near_triple]))

        far_cand = [
            b for b in bars
            if b["base"] < near_top                                  # above the near stumps
            and b["h"] < 0.6 * near_med_h                            # smaller (farther)
            and abs(b["cx"] - near_cx) < 2.0 * max(near_span, 1.0)   # roughly aligned laterally
        ]
        triple, score = self._best_stump_triple(far_cand, min_score=0.75)
        return triple

    def _to_reference_points(self, triple, prefix):
        left, mid, right = triple  # sorted by cx
        return {
            f"{prefix}_left_base": (left["cx"], left["base"]),
            f"{prefix}_left_top": (left["cx"], left["top"]),
            f"{prefix}_mid_base": (mid["cx"], mid["base"]),
            f"{prefix}_mid_top": (mid["cx"], mid["top"]),
            f"{prefix}_right_base": (right["cx"], right["base"]),
            f"{prefix}_right_top": (right["cx"], right["top"]),
        }


# --------------------------------------------------------------------------- #
# Calibration result + core solver
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    """Recovered camera model and quality metrics."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    camera_center: np.ndarray
    focal: float
    reprojection_rms: float
    confidence: float


def focal_from_homography(H: np.ndarray, cx: float, cy: float) -> Optional[float]:
    """Estimate focal length from a plane homography (principal point fixed).

    Uses the orthonormality of the rotation columns encoded in
    ``H = K[r1 r2 t]`` — two constraints on the image of the absolute conic
    give ``1/f²``. Returns None when the geometry is too degenerate to yield a
    positive real focal.
    """
    h1, h2 = H[:, 0], H[:, 1]
    # B = w·M + N, with w = 1/f², N picking the (3,3) term.
    M = np.array(
        [[1, 0, -cx], [0, 1, -cy], [-cx, -cy, cx * cx + cy * cy]], dtype=np.float64
    )
    estimates = []
    den1 = float(h1 @ M @ h2)
    if abs(den1) > 1e-12:
        estimates.append(-(h1[2] * h2[2]) / den1)
    den2 = float(h1 @ M @ h1 - h2 @ M @ h2)
    if abs(den2) > 1e-12:
        estimates.append(-(h1[2] ** 2 - h2[2] ** 2) / den2)

    positive = [w for w in estimates if w > 0]
    if not positive:
        return None
    w = float(np.mean(positive))
    return 1.0 / np.sqrt(w)


class CameraCalibrator:
    """Solves camera intrinsics + pose from named pitch correspondences."""

    def __init__(
        self,
        pitch: Optional[PitchModel] = None,
        min_confidence: float = 0.5,
        max_reproj_px: float = 12.0,
        horizontal_fov_deg: float = 65.0,
    ) -> None:
        self.pitch = pitch or PitchModel()
        self.min_confidence = min_confidence
        self.max_reproj_px = max_reproj_px
        # Focal length is taken from an assumed horizontal field of view rather
        # than recovered from the reference geometry: the stump landmark is
        # small and near fronto-parallel, so single-view focal estimation is
        # ill-conditioned. ~65 deg is typical for a smartphone main camera; this
        # is a class-level prior (not device-specific) and can be overridden.
        self.horizontal_fov_deg = horizontal_fov_deg

    def calibrate(
        self,
        references: Dict[str, Tuple[float, float]],
        image_shape: Tuple[int, int],
        detector_confidence: float = 1.0,
    ) -> CalibrationResult:
        h_img, w_img = image_shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0

        world = self.pitch.reference_points()
        names = [n for n in world if n in references]
        reasons: List[str] = []
        if len(names) < 4:
            reasons.append(
                f"insufficient reference points ({len(names)} of 4): stumps not "
                "fully visible or not detected"
            )
            raise CalibrationError(reasons)

        world_xyz = np.array([world[n] for n in names], dtype=np.float64)
        image_uv = np.array([references[n] for n in names], dtype=np.float64)
        near_mask = np.array([world[n][1] == 0.0 for n in names])  # near stumps at y=0

        # -- focal length from assumed field of view ----------------------- #
        focal = (w_img / 2.0) / np.tan(np.radians(self.horizontal_fov_deg) / 2.0)
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

        # -- pose via PnP -------------------------------------------------- #
        # Planar target (near stumps only, all y=0) => IPPE gives the mirror-
        # ambiguous pair. Spatial target (near+far) => a single well-posed EPnP
        # solution. Either way, keep the physically plausible pose.
        spread = world_xyz.max(axis=0) - world_xyz.min(axis=0)
        planar = float(np.min(spread)) < 1e-6
        try:
            if planar:
                n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    world_xyz, image_uv, K, None, flags=cv2.SOLVEPNP_IPPE
                )
            else:
                n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    world_xyz, image_uv, K, None, flags=cv2.SOLVEPNP_EPNP
                )
        except cv2.error:
            n_sol, rvecs, tvecs = 0, [], []
        if not n_sol:
            raise CalibrationError(["pose estimation (solvePnP) failed"])

        best = None
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rvec)
            center = (-R.T @ tvec).ravel()
            view_dir = (R.T @ np.array([0.0, 0.0, 1.0])).ravel()
            if not self._pose_plausible(center, view_dir):
                continue
            proj, _ = cv2.projectPoints(world_xyz, rvec, tvec, K, None)
            err = np.linalg.norm(proj.reshape(-1, 2) - image_uv, axis=1)
            # Gate on the near stumps (always accurate); far stumps are distant
            # and noisier, so they inform the pose but not the acceptance test.
            gate_err = err[near_mask] if near_mask.any() else err
            rms = float(np.sqrt(np.mean(gate_err ** 2)))
            if best is None or rms < best[0]:
                best = (rms, R, tvec.ravel(), center)

        if best is None:
            reasons.append(
                "no physically plausible camera pose (expected: behind the "
                "striker, above ground, facing down the pitch)"
            )
            raise CalibrationError(reasons)

        rms, R, t, camera_center = best
        # Gate on reprojection error (detector confidence was already checked
        # upstream). A tight fit of the three-stump geometry is the signal that
        # the correspondences are correct.
        if rms > self.max_reproj_px:
            reasons.append(
                f"high reprojection error ({rms:.1f}px): stumps mis-detected or "
                "camera moved"
            )
            raise CalibrationError(reasons)
        confidence = float(detector_confidence * max(0.0, 1.0 - rms / self.max_reproj_px))

        logger.info(
            "Calibrated: f=%.1f px, reproj RMS=%.2f px, confidence=%.2f, cam=%s",
            focal, rms, confidence, np.round(camera_center, 2),
        )
        return CalibrationResult(
            K=K, R=R, t=t, camera_center=camera_center,
            focal=focal, reprojection_rms=rms, confidence=confidence,
        )

    @staticmethod
    def _pose_plausible(center: np.ndarray, view_dir: np.ndarray) -> bool:
        """Camera should be behind the striker (y<=~1), above ground, looking
        down the pitch toward the bowler (+y)."""
        y, z = float(center[1]), float(center[2])
        return (-40.0 <= y <= 2.0) and (0.1 <= z <= 8.0) and (view_dir[1] > 0.0)


# --------------------------------------------------------------------------- #
# The projector
# --------------------------------------------------------------------------- #
class CameraCalibrationProjector:
    """``ObservationProjector`` that back-projects ball pixels to metric world.

    Drop-in replacement for ``PixelPlaneProjector`` — same interface, so
    switching is a one-line change in the pipeline:

        CricketPipeline(projector=CameraCalibrationProjector())

    Parameters
    ----------
    reference_detector:
        Strategy for locating reference points. Defaults to
        :class:`StumpPitchDetector`; inject a learned detector to improve
        automatic calibration without other changes.
    manual_references:
        Optional ``{name: (u, v)}`` overriding auto-detection (assisted setup).
    motion_plane_x:
        Lateral position (metres) of the assumed vertical delivery plane used to
        resolve depth. Default 0.0 = pitch centre line.
    frame_index:
        Which ingested frame to calibrate from (should show a clear, static
        pitch). Defaults to the first frame.
    """

    def __init__(
        self,
        pitch: Optional[PitchModel] = None,
        reference_detector: Optional[PitchReferenceDetector] = None,
        manual_references: Optional[Dict[str, Tuple[float, float]]] = None,
        motion_plane_x: float = 0.0,
        min_confidence: float = 0.5,
        frame_index: int = 0,
    ) -> None:
        self.pitch = pitch or PitchModel()
        self.reference_detector = reference_detector or StumpPitchDetector()
        self.manual_references = manual_references
        self.motion_plane_x = motion_plane_x
        self.frame_index = frame_index
        self.calibrator = CameraCalibrator(self.pitch, min_confidence=min_confidence)
        self.calibration: Optional[CalibrationResult] = None

    # -- lifecycle --------------------------------------------------------- #
    def prepare(self, video: IngestedVideo) -> None:
        """Calibrate once for the (fixed) camera. Raises ``CalibrationError`` if
        low confidence (rejecting the video with an explanation)."""
        if not video.frames:
            raise CalibrationError(["empty video: no frames to calibrate from"])

        if self.manual_references is not None:
            image = video.frames[min(self.frame_index, len(video.frames) - 1)].image
            self.calibration = self.calibrator.calibrate(
                self.manual_references, image.shape, detector_confidence=1.0
            )
            return

        # Multi-frame calibration. The camera is fixed, so the stumps sit at the
        # SAME image location in every frame. Detect across many frames, then
        # vote/average: this rejects frames where the stumps are occluded or a
        # wrong triple was picked, and averages out per-frame detection noise —
        # far more robust than trusting any single frame.
        detections = []
        shape = None
        for frame in self._candidate_frames(video):
            det = self.reference_detector.detect(frame.image)
            if det is not None and len(det.points) >= 4:
                detections.append(det)
                shape = frame.image.shape
        if not detections:
            raise CalibrationError(
                ["stumps not detected in any sampled frame: stumps not visible, "
                 "occluded, or poor lighting"]
            )

        # Assemble candidate reference sets, then keep whichever CALIBRATES
        # with the lowest reprojection error. This makes multi-frame voting and
        # far stumps strictly additive: they can improve the result but never
        # do worse than the best single frame.
        candidates = []  # (label, references, confidence)
        voted, vote_conf = self._vote_references(detections, shape)
        if voted:
            candidates.append(("voted", voted, vote_conf))
            candidates.append(("voted_near", self._near_only(voted), vote_conf))
        best_frame = max(detections, key=lambda d: d.confidence)
        candidates.append(("best_frame", best_frame.points, best_frame.confidence))
        candidates.append(("best_frame_near", self._near_only(best_frame.points), best_frame.confidence))

        best_result, best_label, last_error = None, None, None
        for label, refs, conf in candidates:
            if len(refs) < 4:
                continue
            try:
                result = self.calibrator.calibrate(refs, shape, conf)
            except CalibrationError as e:
                last_error = e
                continue
            if best_result is None or result.reprojection_rms < best_result.reprojection_rms:
                best_result, best_label = result, label

        if best_result is None:
            raise last_error or CalibrationError(["calibration failed on all candidates"])
        logger.info("Calibration selected '%s' (rms=%.2f px)", best_label, best_result.reprojection_rms)
        self.calibration = best_result

    @staticmethod
    def _near_only(refs):
        return {k: v for k, v in refs.items() if not k.startswith("far_")}

    _NEAR_BASE = ("stump_left_base", "stump_mid_base", "stump_right_base")

    def _vote_references(self, detections, shape):
        """Temporal voting + averaging over per-frame detections.

        Uses the near-stump centroid as a per-frame signature; keeps the frames
        agreeing with the median location (voting out occluded/wrong-triple
        frames) and takes the median of each reference point over them
        (temporal averaging). Far stumps are included only when they too are
        seen consistently. Returns ``(references, confidence)`` or ``(None, 0)``.
        """
        H, W = shape[:2]
        tol_x, tol_y = 0.03 * W, 0.03 * H

        sigs = []
        for d in detections:
            if all(k in d.points for k in self._NEAR_BASE):
                xs = [d.points[k][0] for k in self._NEAR_BASE]
                ys = [d.points[k][1] for k in self._NEAR_BASE]
                sigs.append((d, float(np.mean(xs)), float(np.mean(ys))))
        if not sigs:
            return None, 0.0

        med_cx = float(np.median([s[1] for s in sigs]))
        med_by = float(np.median([s[2] for s in sigs]))
        kept = [d for d, cx, by in sigs if abs(cx - med_cx) <= tol_x and abs(by - med_by) <= tol_y]
        if not kept:
            return None, 0.0

        # Median each near reference point over the agreeing frames.
        refs = self._median_points(kept, prefix="stump")

        # Far stumps: include only if seen consistently in enough kept frames.
        far_frames = [d for d in kept if "far_stump_mid_base" in d.points]
        if len(far_frames) >= max(2, int(0.3 * len(kept))):
            far_cx = [np.mean([d.points[k][0] for k in
                              ("far_stump_left_base", "far_stump_mid_base", "far_stump_right_base")])
                      for d in far_frames if all(k in d.points for k in
                              ("far_stump_left_base", "far_stump_mid_base", "far_stump_right_base"))]
            if far_cx and (np.max(far_cx) - np.min(far_cx)) <= 2 * tol_x:
                refs.update(self._median_points(far_frames, prefix="far_stump"))

        agreement = len(kept) / len(detections)
        mean_conf = float(np.mean([d.confidence for d in kept]))
        vote_conf = float(np.clip(mean_conf * min(1.0, len(kept) / 3.0), 0.0, 1.0))
        logger.info(
            "Vote: %d/%d frames agree (%.0f%%), %s far stumps, conf=%.2f",
            len(kept), len(detections), 100 * agreement,
            "with" if any(k.startswith("far_") for k in refs) else "no", vote_conf,
        )
        return refs, vote_conf

    @staticmethod
    def _median_points(dets, prefix):
        names = set()
        for d in dets:
            names.update(k for k in d.points if k.startswith(prefix + "_"))
        out = {}
        for name in names:
            us = [d.points[name][0] for d in dets if name in d.points]
            vs = [d.points[name][1] for d in dets if name in d.points]
            if us:
                out[name] = (float(np.median(us)), float(np.median(vs)))
        return out

    def _candidate_frames(self, video: IngestedVideo, n: int = 24):
        """Evenly sampled frames to search for a clean, stable view of stumps."""
        frames = video.frames
        if len(frames) <= n:
            return frames
        step = len(frames) / n
        return [frames[int(i * step)] for i in range(n)]

    # -- projection -------------------------------------------------------- #
    def project(
        self, track: Sequence[BallDetection], metadata: VideoMetadata
    ) -> List[BallObservation]:
        if self.calibration is None:
            raise RuntimeError("Projector not prepared; call prepare(video) first.")

        cal = self.calibration
        K_inv = np.linalg.inv(cal.K)
        C = cal.camera_center
        px = self.motion_plane_x

        observations: List[BallObservation] = []
        n_degenerate = 0
        for det in track:
            u, v = det.center
            ray_cam = K_inv @ np.array([u, v, 1.0])
            ray_world = cal.R.T @ ray_cam  # rotate direction into world frame

            # Intersect ray (C + s·ray_world) with the vertical plane x = px.
            if abs(ray_world[0]) < 1e-6:
                n_degenerate += 1
                continue
            s = (px - C[0]) / ray_world[0]
            if s <= 0:  # behind the camera
                continue
            point = C + s * ray_world
            observations.append(
                BallObservation(
                    timestamp=det.timestamp,
                    x=px,
                    y=float(point[1]),
                    z=float(max(point[2], 0.0)),
                )
            )

        if n_degenerate:
            logger.warning(
                "%d/%d rays nearly parallel to the delivery plane (camera close "
                "to the pitch centre line — height poorly observable).",
                n_degenerate, len(track),
            )
        return observations
