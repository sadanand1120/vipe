from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.parse

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PROGRAM_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROGRAM_DIR.parent
for import_dir in (PROGRAM_DIR, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from get_best_view import (
    DEFAULT_RAW_ROOT,
    DEFAULT_SCENE_DIR,
    build_instance_samples,
    evaluate_view,
    load_color_intrinsics,
    load_frames,
    load_instance_ply,
    project_points,
)
from view_pcd import DEFAULT_HOST, DEFAULT_PCD_DIR


DEFAULT_PORT = 8092


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manual Best View Editor</title>
  <style>
    html, body { margin: 0; min-height: 100%; background: #111; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; }
    #wrap { padding: 14px; display: grid; gap: 14px; grid-template-columns: 1fr 1fr; }
    .panel { background: #181818; border: 1px solid #444; border-radius: 10px; padding: 12px; min-width: 0; }
    .panel h2 { margin: 0 0 10px; font-size: 16px; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 10px; }
    input, select, button { color: #eee; background: #222; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    input[type="range"] { flex: 1; min-width: 260px; }
    button { cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    img { display: block; max-width: 100%; max-height: calc(100vh - 170px); object-fit: contain; border: 1px solid #333; background: #000; }
    #status { position: sticky; bottom: 0; grid-column: 1 / -1; padding: 8px 10px; background: rgba(20,20,20,.94); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
    .muted { color: #bbb; }
  </style>
</head>
<body>
  <div id="wrap">
    <section class="panel">
      <h2>Sequence Images</h2>
      <div class="row">
        <button id="prevFrameBtn">Previous</button>
        <input id="frameSlider" type="range" min="0" max="0" value="0" />
        <button id="nextFrameBtn">Next</button>
        <button id="playBtn">Play</button>
        <button id="stopBtn" disabled>Stop</button>
        <span id="frameLabel"></span>
      </div>
      <img id="frameImg" alt="sequence frame" />
    </section>
    <section class="panel">
      <h2>Instance Best View</h2>
      <div class="row">
        <button id="prevInstanceBtn">Previous</button>
        <label>Instance <select id="instanceSelect"></select></label>
        <button id="nextInstanceBtn">Next</button>
        <span id="instanceLabel" class="muted"></span>
      </div>
      <div class="row">
        <label>Image id <input id="imageIdInput" type="text" size="8" /></label>
        <button id="previewBtn">Preview</button>
        <button id="useCurrentBtn">Use Top Image</button>
        <button id="saveBtn">Save</button>
      </div>
      <img id="previewImg" alt="best-view preview" />
    </section>
    <div id="status">Loading...</div>
  </div>
  <script>
    const frameSlider = document.getElementById('frameSlider');
    const prevFrameBtn = document.getElementById('prevFrameBtn');
    const nextFrameBtn = document.getElementById('nextFrameBtn');
    const playBtn = document.getElementById('playBtn');
    const stopBtn = document.getElementById('stopBtn');
    const frameLabel = document.getElementById('frameLabel');
    const frameImg = document.getElementById('frameImg');
    const instanceSelect = document.getElementById('instanceSelect');
    const prevInstanceBtn = document.getElementById('prevInstanceBtn');
    const nextInstanceBtn = document.getElementById('nextInstanceBtn');
    const instanceLabel = document.getElementById('instanceLabel');
    const imageIdInput = document.getElementById('imageIdInput');
    const previewBtn = document.getElementById('previewBtn');
    const useCurrentBtn = document.getElementById('useCurrentBtn');
    const saveBtn = document.getElementById('saveBtn');
    const previewImg = document.getElementById('previewImg');
    const statusEl = document.getElementById('status');

    let meta = null;
    let playTimer = null;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function currentFrame() {
      return meta.frames[Number(frameSlider.value)];
    }

    function currentInstance() {
      return meta.instances.find((item) => String(item.instance_id) === String(instanceSelect.value));
    }

    function updateFrame() {
      const frame = currentFrame();
      frameLabel.textContent = `${Number(frameSlider.value) + 1}/${meta.frames.length}: image_id=${frame.image_id}`;
      frameImg.src = `/frame?idx=${encodeURIComponent(frameSlider.value)}&t=${Date.now()}`;
      prevFrameBtn.disabled = Number(frameSlider.value) <= 0;
      nextFrameBtn.disabled = Number(frameSlider.value) >= meta.frames.length - 1;
    }

    function stopPlayback() {
      if (playTimer !== null) {
        clearInterval(playTimer);
        playTimer = null;
      }
      playBtn.disabled = false;
      stopBtn.disabled = true;
    }

    function playFromCurrent() {
      stopPlayback();
      playBtn.disabled = true;
      stopBtn.disabled = false;
      playTimer = setInterval(() => {
        const next = Number(frameSlider.value) + 1;
        if (next >= meta.frames.length) {
          stopPlayback();
          return;
        }
        frameSlider.value = String(next);
        updateFrame();
      }, 100);
    }

    function updateInstance() {
      const inst = currentInstance();
      const label = inst.semantic_label ? `${inst.semantic_label}, ` : '';
      instanceLabel.textContent = `${label}current=${inst.current_image_id || 'none'}`;
      imageIdInput.value = inst.current_image_id || currentFrame().image_id;
      prevInstanceBtn.disabled = instanceSelect.selectedIndex <= 0;
      nextInstanceBtn.disabled = instanceSelect.selectedIndex >= instanceSelect.options.length - 1;
      updatePreview();
    }

    function updatePreview() {
      const instanceId = instanceSelect.value;
      const imageId = imageIdInput.value.trim();
      if (!imageId) {
        setStatus('Enter an image id.');
        return;
      }
      previewImg.src = `/render?instance_id=${encodeURIComponent(instanceId)}&image_id=${encodeURIComponent(imageId)}&t=${Date.now()}`;
      setStatus(`Preview instance ${instanceId} on image ${imageId}`);
    }

    async function saveBestView() {
      const instanceId = Number(instanceSelect.value);
      const imageId = imageIdInput.value.trim();
      saveBtn.disabled = true;
      setStatus(`Saving instance ${instanceId} -> image ${imageId}...`);
      try {
        const response = await fetch('/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({instance_id: instanceId, image_id: imageId}),
        });
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        const inst = currentInstance();
        inst.current_image_id = payload.best_view.image_id;
        inst.current_image_path = payload.best_view.image_path;
        instanceLabel.textContent = `${inst.semantic_label ? inst.semantic_label + ', ' : ''}current=${inst.current_image_id}`;
        updatePreview();
        setStatus(`Saved instance ${instanceId} -> image ${payload.best_view.image_id}; JSON and PNG updated.`);
      } catch (err) {
        console.error(err);
        setStatus(`Save failed: ${err.message || err}`);
      } finally {
        saveBtn.disabled = false;
      }
    }

    async function init() {
      const response = await fetch('/meta');
      meta = await response.json();
      frameSlider.max = String(meta.frames.length - 1);
      frameSlider.value = '0';
      for (const inst of meta.instances) {
        const option = document.createElement('option');
        option.value = String(inst.instance_id);
        option.textContent = inst.semantic_label ? `${inst.instance_id} (${inst.semantic_label})` : String(inst.instance_id);
        instanceSelect.appendChild(option);
      }
      updateFrame();
      updateInstance();
      setStatus(`Loaded ${meta.frames.length} frames and ${meta.instances.length} instances.`);
    }

    frameSlider.addEventListener('input', updateFrame);
    prevFrameBtn.addEventListener('click', () => {
      frameSlider.value = String(Math.max(0, Number(frameSlider.value) - 1));
      updateFrame();
    });
    nextFrameBtn.addEventListener('click', () => {
      frameSlider.value = String(Math.min(meta.frames.length - 1, Number(frameSlider.value) + 1));
      updateFrame();
    });
    playBtn.addEventListener('click', playFromCurrent);
    stopBtn.addEventListener('click', stopPlayback);
    instanceSelect.addEventListener('change', updateInstance);
    prevInstanceBtn.addEventListener('click', () => {
      instanceSelect.selectedIndex = Math.max(0, instanceSelect.selectedIndex - 1);
      updateInstance();
    });
    nextInstanceBtn.addEventListener('click', () => {
      instanceSelect.selectedIndex = Math.min(instanceSelect.options.length - 1, instanceSelect.selectedIndex + 1);
      updateInstance();
    });
    previewBtn.addEventListener('click', updatePreview);
    imageIdInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') updatePreview();
    });
    useCurrentBtn.addEventListener('click', () => {
      imageIdInput.value = currentFrame().image_id;
      updatePreview();
    });
    saveBtn.addEventListener('click', saveBestView);
    init().catch((err) => {
      console.error(err);
      setStatus(`Init failed: ${err.message || err}`);
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a manual best-view editor.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--best-views-json", type=Path, default=None)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--sample-points-per-instance", type=int, default=None)
    parser.add_argument("--min-visible-points", type=int, default=None)
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


class ManualBestViewState:
    def __init__(
        self,
        pcd_dir: Path,
        scene_dir: Path,
        raw_root: Path,
        best_views_json: Path,
        sample_points_per_instance: int | None,
        min_visible_points: int | None,
        min_depth: float | None,
        seed: int | None,
    ) -> None:
        self.pcd_dir = pcd_dir
        self.scene_dir = scene_dir
        self.raw_root = raw_root
        self.best_views_json = best_views_json
        self.data = json.loads(best_views_json.read_text())
        params = self.data.get("params", {})
        self.sample_points_per_instance = int(sample_points_per_instance or params.get("sample_points_per_instance", 4096))
        self.min_visible_points = int(min_visible_points or params.get("min_visible_points", 25))
        self.min_depth = float(min_depth if min_depth is not None else params.get("min_depth", 0.05))
        self.seed = int(seed if seed is not None else params.get("seed", 42))
        self.image_output_dir = Path(self.data.get("best_view_image_dir") or str(pcd_dir / "best_views")).expanduser().resolve()

        self.intrinsics, self.width, self.height = load_color_intrinsics(scene_dir, raw_root)
        points, point_instance_ids = load_instance_ply(pcd_dir / "instance.ply")
        self.frames = load_frames(scene_dir, -1)
        self.frames_by_id = {str(frame["image_id"]): frame for frame in self.frames}
        self.instance_ids, self.samples = build_instance_samples(points, point_instance_ids, self.sample_points_per_instance, self.seed)

    def meta(self) -> dict[str, object]:
        instances = []
        for instance_id_text, item in sorted(self.data["instances"].items(), key=lambda kv: int(kv[0])):
            best = item.get("best_view", {})
            instances.append(
                {
                    "instance_id": int(instance_id_text),
                    "semantic_label": item.get("semantic_label", ""),
                    "current_image_id": best.get("image_id", ""),
                    "current_image_path": best.get("image_path", ""),
                }
            )
        return {
            "frames": [
                {"idx": idx, "image_id": frame["image_id"], "image_path": frame["image_path"]}
                for idx, frame in enumerate(self.frames)
            ],
            "instances": instances,
        }

    def frame_path(self, idx: int) -> Path:
        if idx < 0 or idx >= len(self.frames):
            raise IndexError(f"frame idx out of range: {idx}")
        return Path(str(self.frames[idx]["image_path"]))

    def render_preview(self, instance_id: int, image_id: str, output_path: Path | None = None) -> tuple[bytes, dict[str, object]]:
        import cv2

        frame = self.frames_by_id.get(str(image_id))
        if frame is None:
            raise ValueError(f"Unknown image id: {image_id}")
        if int(instance_id) not in self.samples:
            raise ValueError(f"Unknown instance id: {instance_id}")

        points = self.samples[int(instance_id)]["points"]
        metrics = evaluate_view(
            points,
            frame,
            self.intrinsics,
            self.width,
            self.height,
            self.min_depth,
            self.min_visible_points,
        )
        if metrics is None:
            raise ValueError(f"Image {image_id} has too few visible points for instance {instance_id}")

        bgr = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {frame['image_path']}")
        overlay = bgr.copy()
        u, v, z = project_points(points, frame["w2c"], self.intrinsics)
        valid = (z > self.min_depth) & (u >= 0.0) & (u < self.width) & (v >= 0.0) & (v < self.height)
        uv = np.column_stack([u[valid], v[valid]]).round().astype(np.int32)
        for x, y in uv:
            cv2.circle(overlay, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        xmin, ymin, xmax, ymax = [int(round(x)) for x in metrics["bbox_xyxy"]]
        cv2.rectangle(
            overlay,
            (max(0, xmin), max(0, ymin)),
            (min(self.width - 1, xmax), min(self.height - 1, ymax)),
            color=(0, 0, 255),
            thickness=4,
            lineType=cv2.LINE_AA,
        )

        title_height = 58
        gutter = 18
        canvas = np.full((self.height + title_height, self.width * 2 + gutter, 3), 255, dtype=np.uint8)
        canvas[title_height:, : self.width] = bgr
        canvas[title_height:, self.width + gutter :] = overlay
        canvas[:, self.width : self.width + gutter] = np.array([255, 255, 255], dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_title = "RGB"
        right_title = "Projected points + bbox"
        left_size = cv2.getTextSize(left_title, font, 1.1, 2)[0]
        right_size = cv2.getTextSize(right_title, font, 1.1, 2)[0]
        cv2.putText(canvas, left_title, ((self.width - left_size[0]) // 2, 38), font, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            right_title,
            (self.width + gutter + (self.width - right_size[0]) // 2, 38),
            font,
            1.1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise OSError("Could not encode preview PNG")
        png = encoded.tobytes()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(png)
        return png, metrics

    def save(self, instance_id: int, image_id: str) -> dict[str, object]:
        output_path = self.image_output_dir / f"{int(instance_id)}.png"
        _, metrics = self.render_preview(instance_id, image_id, output_path)
        frame = self.frames_by_id[str(image_id)]
        best_view = {
            "found": True,
            "image_id": frame["image_id"],
            "image_path": frame["image_path"],
            "pose_path": frame["pose_path"],
            **metrics,
            "best_view_image_path": str(output_path),
        }
        instance_key = str(int(instance_id))
        self.data["instances"][instance_key]["best_view"] = best_view
        with self.best_views_json.open("w") as f:
            json.dump(self.data, f, indent=2)
        return best_view


def make_handler(state: ManualBestViewState):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

        def send_plain_error(self, status: int, message: str) -> None:
            data = message.encode("utf-8", errors="replace")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    data = HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if parsed.path == "/meta":
                    self.send_json(state.meta())
                    return
                if parsed.path == "/frame":
                    path = state.frame_path(int(query["idx"][0]))
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(path.stat().st_size))
                    self.end_headers()
                    with path.open("rb") as f:
                        self.wfile.write(f.read())
                    return
                if parsed.path == "/render":
                    png, _ = state.render_preview(int(query["instance_id"][0]), str(query["image_id"][0]))
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(png)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(png)
                    return
            except Exception as exc:
                self.send_plain_error(500, str(exc))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/save":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                best_view = state.save(int(request["instance_id"]), str(request["image_id"]).strip())
                self.send_json({"best_view": best_view})
            except Exception as exc:
                self.send_plain_error(500, str(exc))

    return Handler


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    scene_dir = args.scene_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    best_views_json = args.best_views_json.expanduser().resolve() if args.best_views_json else pcd_dir / "best_views.json"
    state = ManualBestViewState(
        pcd_dir,
        scene_dir,
        raw_root,
        best_views_json,
        args.sample_points_per_instance,
        args.min_visible_points,
        args.min_depth,
        args.seed,
    )
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
    print(f"Serving manual best-view editor for {pcd_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print("Stop with Ctrl-C")
    server.serve_forever()


if __name__ == "__main__":
    main()
